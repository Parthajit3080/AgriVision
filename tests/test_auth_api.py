import os
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import httpx

ORIGIN = "http://localhost:3000"
TEST_SECRET = "test-secret-for-agrivision-account-tests-32-bytes-minimum"


class AccountApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".auth-test-", dir=os.getcwd())
        self.env = ExitStack()
        self.env.enter_context(patch.dict(os.environ, {
            "APP_ENV": "test",
            "AGRIVISION_SECRET": TEST_SECRET,
            "AGRIVISION_DB_PATH": os.path.join(self.temporary.name, "accounts.sqlite3"),
            "AGRIVISION_COOKIE_SECURE": "false",
            "AGRIVISION_COOKIE_SAMESITE": "lax",
            "FRONTEND_ORIGIN": ORIGIN,
            "APP_PUBLIC_URL": ORIGIN,
            "SMTP_HOST": "",
        }, clear=False))
        from backend.app import app
        self.app = app
        self.lifespan = app.router.lifespan_context(app)
        await self.lifespan.__aenter__()
        self.transport = httpx.ASGITransport(app=app)
        self.client = httpx.AsyncClient(transport=self.transport, base_url="http://testserver")

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.lifespan.__aexit__(None, None, None)
        self.env.close()
        self.temporary.cleanup()

    async def csrf(self, client=None):
        client = client or self.client
        response = await client.get("/api/auth/csrf")
        self.assertEqual(response.status_code, 200)
        return response.json()["csrf_token"]

    async def mutate(self, path, *, method="POST", body=None, client=None, csrf=True, origin=ORIGIN):
        client = client or self.client
        headers = {"Origin": origin}
        if csrf:
            headers["X-CSRF-Token"] = await self.csrf(client)
        return await client.request(method, path, json=body, headers=headers)

    async def signup(self, client=None, *, email="farmer@example.com", display_name="Field Farmer", password="a-long-test-password-2026"):
        return await self.mutate("/api/auth/signup", body={"email": email, "display_name": display_name, "password": password}, client=client)

    async def test_password_hashes_are_salted_and_verifiable(self):
        from backend import auth_store
        first = auth_store.hash_password("a-long-test-password-2026")
        second = auth_store.hash_password("a-long-test-password-2026")
        self.assertNotEqual(first, second)
        self.assertTrue(auth_store.verify_password("a-long-test-password-2026", first))
        self.assertFalse(auth_store.verify_password("a-different-password-2026", first))

    async def test_guest_prediction_remains_public_and_accounts_can_save_reports(self):
        from backend import app as api_module
        from backend import auth_store

        inputs = {
            "location": {"state": "Punjab", "district": "Ludhiana"},
            "season": "Kharif", "land_size": 2, "land_unit": "acres",
            "soil_test": {"N": 70, "P": 45, "K": 45, "ph": 6.5},
            "preferred_fertilizers": [], "excluded_fertilizers": [],
            "risk_tolerance": "medium", "target_year": 2027, "optimization_goal": "balance_roi",
        }
        legacy_response = {"legacy_contract": {"recommendation": "example"}}
        with patch.object(api_module.model, "predict", return_value=legacy_response):
            response = await self.client.post("/api/predict", json=inputs)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), legacy_response)
            with auth_store.db_session() as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM planner_reports").fetchone()[0], 0)

            created = await self.signup()
            self.assertEqual(created.status_code, 201, created.text)
            self.assertTrue(self.client.cookies.get("agrivision_session"))
            response = await self.client.post("/api/predict", json=inputs, headers={"Origin": ORIGIN, "X-CSRF-Token": await self.csrf()})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json(), legacy_response)
            history = await self.client.get("/api/auth/history")
            self.assertEqual(history.status_code, 200)
            self.assertEqual(len(history.json()["items"]), 1)
            self.assertEqual(history.json()["items"][0]["district"], "Ludhiana")

    async def test_signup_profile_preferences_sessions_and_logout(self):
        response = await self.signup()
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["user"]["email"], "farmer@example.com")
        self.assertNotIn("password_hash", response.json()["user"])
        self.assertIn("httponly", response.headers.get("set-cookie", "").lower())
        self.assertEqual((await self.client.get("/api/auth/me")).status_code, 200)

        profile = await self.mutate("/api/auth/profile", method="PATCH", body={"display_name": "Asha Rao"})
        self.assertEqual(profile.status_code, 200, profile.text)
        self.assertEqual(profile.json()["user"]["display_name"], "Asha Rao")
        preferences = await self.mutate("/api/auth/preferences", method="PUT", body={
            "form": {"season": "Rabi", "land_size": "3"}, "preferred": ["Urea"], "excluded": [],
        })
        self.assertEqual(preferences.status_code, 200, preferences.text)
        sessions = (await self.client.get("/api/auth/sessions")).json()["sessions"]
        self.assertEqual(len(sessions), 1)
        self.assertTrue(sessions[0]["current"])
        self.assertEqual((await self.mutate("/api/auth/logout", body={})).status_code, 204)
        self.assertEqual((await self.client.get("/api/auth/me")).status_code, 401)

    async def test_origin_and_csrf_are_required_for_state_changes(self):
        body = {"email": "x@example.com", "display_name": "X", "password": "another-long-test-password"}
        response = await self.mutate("/api/auth/signup", body=body, csrf=False)
        self.assertEqual(response.status_code, 403)
        response = await self.mutate("/api/auth/signup", body=body, origin="https://attacker.example")
        self.assertEqual(response.status_code, 403)

    async def test_failed_logins_persist_rate_limit_attempts(self):
        from backend import auth_store
        with patch.object(auth_store, "verify_password", return_value=False):
            for attempt in range(12):
                response = await self.mutate("/api/auth/login", body={"email": "unknown@example.com", "password": "incorrect"})
                self.assertEqual(response.status_code, 401, f"attempt {attempt + 1}: {response.text}")
            blocked = await self.mutate("/api/auth/login", body={"email": "unknown@example.com", "password": "incorrect"})
        self.assertEqual(blocked.status_code, 429)
        with auth_store.db_session() as connection:
            saved = connection.execute("SELECT COUNT(*) FROM auth_attempts WHERE scope = 'login'").fetchone()[0]
            self.assertEqual(saved, 12)

    async def test_password_recovery_is_single_use_and_revokes_sessions(self):
        self.assertEqual((await self.signup()).status_code, 201)
        response = await self.mutate("/api/auth/forgot-password", body={"email": "farmer@example.com"})
        self.assertEqual(response.status_code, 200, response.text)
        reset_url = response.json()["development_reset_url"]
        token = parse_qs(urlparse(reset_url).query)["token"][0]
        reset = await self.mutate("/api/auth/reset-password", body={"token": token, "password": "a-new-secure-password-2026"})
        self.assertEqual(reset.status_code, 200, reset.text)
        self.assertEqual((await self.client.get("/api/auth/me")).status_code, 401)
        second_use = await self.mutate("/api/auth/reset-password", body={"token": token, "password": "another-new-password-2026"})
        self.assertEqual(second_use.status_code, 400)
        login = await self.mutate("/api/auth/login", body={"email": "farmer@example.com", "password": "a-new-secure-password-2026"})
        self.assertEqual(login.status_code, 200, login.text)

    async def test_membership_review_rbac_and_immutable_admin_audit(self):
        from backend import auth_store

        self.assertEqual((await self.signup()).status_code, 201)
        request = await self.mutate("/api/membership/requests", body={"plan_code": "field-plus", "message": "I plan seasonal field reviews."})
        self.assertEqual(request.status_code, 201, request.text)
        application_id = request.json()["application"]["id"]
        self.assertEqual((await self.client.get("/api/premium/analytics")).status_code, 403)
        self.assertEqual((await self.client.get("/api/admin/dashboard")).status_code, 403)

        now = auth_store.timestamp()
        with auth_store.db_session() as connection:
            cursor = connection.execute(
                "INSERT INTO users(email, username, display_name, password_hash, role, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'superadmin', ?, ?)",
                ("admin@example.com", "root-admin", "Platform Admin", auth_store.hash_password("a-very-long-admin-password-2026"), now, now),
            )
            admin_id = cursor.lastrowid

        admin_client = httpx.AsyncClient(transport=self.transport, base_url="http://testserver")
        try:
            login = await self.mutate("/api/admin/auth/login", body={"username": "root-admin", "password": "a-very-long-admin-password-2026"}, client=admin_client)
            self.assertEqual(login.status_code, 200, login.text)
            pending = await admin_client.get("/api/admin/membership-requests?status=pending")
            self.assertEqual(pending.status_code, 200)
            self.assertEqual(len(pending.json()["items"]), 1)
            approval = await self.mutate(f"/api/admin/membership-requests/{application_id}/decision", body={"decision": "approved", "note": "Approved."}, client=admin_client)
            self.assertEqual(approval.status_code, 200, approval.text)
            self.assertEqual((await self.client.get("/api/membership")).json()["status"], "approved")
            self.assertEqual((await self.client.get("/api/premium/analytics")).status_code, 200)
            self.assertEqual((await admin_client.get("/api/admin/dashboard")).json()["metrics"]["premium_approved"], 1)
            searched = await admin_client.get("/api/admin/users?q=farmer%40example.com")
            self.assertEqual(searched.status_code, 200)
            target_id = searched.json()["items"][0]["id"]
            promoted = await self.mutate(f"/api/admin/users/{target_id}/role", method="PATCH", body={"role": "admin"}, client=admin_client)
            self.assertEqual(promoted.status_code, 200, promoted.text)
            self.assertEqual((await self.client.get("/api/auth/me")).status_code, 401)
            relogin = await self.mutate("/api/auth/login", body={"email": "farmer@example.com", "password": "a-long-test-password-2026"})
            self.assertEqual(relogin.status_code, 200, relogin.text)
            self.assertEqual(relogin.json()["user"]["role"], "admin")
            self.assertEqual((await self.client.get("/api/admin/dashboard")).status_code, 200)
            suspended = await self.mutate(f"/api/admin/users/{target_id}/status", method="PATCH", body={"status": "suspended"}, client=admin_client)
            self.assertEqual(suspended.status_code, 200, suspended.text)
            self.assertEqual((await self.client.get("/api/auth/me")).status_code, 401)
            logs = (await admin_client.get("/api/admin/audit-logs")).json()["items"]
            actions = {entry["action"] for entry in logs}
            self.assertTrue({"membership_approved", "user_role_changed", "user_suspended"}.issubset(actions))
            with auth_store.db_session() as connection:
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute("UPDATE admin_audit_logs SET action = 'tampered' WHERE actor_user_id = ?", (admin_id,))
        finally:
            await admin_client.aclose()


if __name__ == "__main__":
    unittest.main()
