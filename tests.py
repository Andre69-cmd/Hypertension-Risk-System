"""Week 8 — Basic unit tests for the Hypertension Risk System."""
import unittest
import os
os.environ['SECRET_KEY'] = 'test-key'

from app import app, db, User, Assessment, Appointment, Reminder, bcrypt


class HypertensionRiskSystemTests(unittest.TestCase):
    """Test core system functionality."""

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test_week8.db'
        app.config['WTF_CSRF_ENABLED'] = False

    def setUp(self):
        with app.app_context():
            db.create_all()
            # Create a test user
            if not User.query.filter_by(email='test@test.com').first():
                u = User(email='test@test.com',
                         password=bcrypt.generate_password_hash('test123').decode('utf-8'))
                db.session.add(u); db.session.commit()

    def tearDown(self):
        with app.app_context():
            db.session.remove()

    # ---------- Basic Tests ----------
    def test_home_page_loads(self):
        client = app.test_client()
        resp = client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'HypertensionRisk', resp.data)

    def test_health_endpoint(self):
        client = app.test_client()
        resp = client.get('/health')
        self.assertIn(resp.status_code, [200, 503])
        data = resp.get_json()
        self.assertIn('status', data)
        self.assertIn('version', data)

    def test_about_page(self):
        client = app.test_client()
        resp = client.get('/about')
        self.assertEqual(resp.status_code, 200)

    def test_login_page_loads(self):
        client = app.test_client()
        resp = client.get('/login')
        self.assertEqual(resp.status_code, 200)

    def test_register_page_loads(self):
        client = app.test_client()
        resp = client.get('/register')
        self.assertEqual(resp.status_code, 200)

    # ---------- Auth Tests ----------
    def test_login_success(self):
        client = app.test_client()
        resp = client.post('/login', data={
            'email': 'test@test.com', 'password': 'test123'
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)

    def test_login_failure(self):
        client = app.test_client()
        resp = client.post('/login', data={
            'email': 'test@test.com', 'password': 'wrongpassword'
        }, follow_redirects=True)
        self.assertIn(b'Invalid', resp.data)

    # ---------- Model Tests ----------
    def test_risk_level_classification(self):
        from app import get_risk_level
        self.assertEqual(get_risk_level(0.1), 'Low')
        self.assertEqual(get_risk_level(0.4), 'Medium')
        self.assertEqual(get_risk_level(0.8), 'High')

    def test_recommendation_lookup(self):
        from app import get_recommendation
        self.assertIn('Great', get_recommendation('Low'))
        self.assertIn('Moderate', get_recommendation('Medium'))
        self.assertIn('High', get_recommendation('High'))

    # ---------- i18n Tests ----------
    def test_translations_exist(self):
        from translations import TRANSLATIONS
        for lang in ['en', 'fr', 'sw']:
            self.assertIn(lang, TRANSLATIONS)
            self.assertIn('app_name', TRANSLATIONS[lang])

    def test_language_switch(self):
        client = app.test_client()
        resp = client.get('/set-language/fr', follow_redirects=True)
        self.assertEqual(resp.status_code, 200)

    # ---------- Security Tests ----------
    def test_protected_routes_redirect(self):
        """Unauthenticated users should be redirected from protected routes."""
        client = app.test_client()
        for route in ['/dashboard', '/assessment', '/appointments', '/reminders']:
            resp = client.get(route)
            self.assertIn(resp.status_code, [302, 401])

    def test_password_hashing(self):
        """Ensure passwords are hashed, not plaintext."""
        with app.app_context():
            user = User.query.filter_by(email='test@test.com').first()
            self.assertNotEqual(user.password, 'test123')
            self.assertTrue(user.password.startswith('$2b$'))


if __name__ == '__main__':
    print("=" * 60)
    print("WEEK 8 — Running Unit Tests")
    print("=" * 60)
    unittest.main(verbosity=2)