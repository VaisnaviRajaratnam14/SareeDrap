import React, { createContext, useContext, useState, useEffect } from 'react';
import api, { getHealth } from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user,         setUser]         = useState(null);
  const [token,        setToken]        = useState(() => localStorage.getItem('saree_token'));
  const [loading,      setLoading]      = useState(true);
  const [dbAvailable,  setDbAvailable]  = useState(true);  // optimistic default

  // Check backend health once on mount
  useEffect(() => {
    getHealth()
      .then(res => setDbAvailable(res.data.database === 'connected'))
      .catch(() => setDbAvailable(false));
  }, []);

  // Re-hydrate user from stored token on mount
  useEffect(() => {
    const init = async () => {
      if (token) {
        try {
          const res = await api.get('/auth/profile');
          setUser(res.data.data);
          setDbAvailable(true);
        } catch (err) {
          // 503 = DB down — clear token silently, don't retry
          if (err.response?.status === 503) {
            setDbAvailable(false);
          }
          localStorage.removeItem('saree_token');
          setToken(null);
        }
      }
      setLoading(false);
    };
    init();
  }, []); // run once only — token change handled below

  const login = async (email, password) => {
    const res = await api.post('/auth/login', { email, password });
    const { token: t, user: u } = res.data.data;
    localStorage.setItem('saree_token', t);
    setToken(t);
    setUser(u);
    setDbAvailable(true);
    return u;
  };

  const register = async (name, email, password) => {
    const res = await api.post('/auth/register', { name, email, password });
    const { token: t, user: u } = res.data.data;
    localStorage.setItem('saree_token', t);
    setToken(t);
    setUser(u);
    setDbAvailable(true);
    return u;
  };

  const logout = () => {
    localStorage.removeItem('saree_token');
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, dbAvailable, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
