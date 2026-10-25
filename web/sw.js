const CACHE_NAME = 'shadowos-v3';
const API_CACHE_NAME = 'shadowos-api-v2';
const API_CACHE_TTL = 300000;
const OFFLINE_QUEUE = 'shadowos-offline-queue';
const MAX_QUEUE_SIZE = 100;

const ASSETS_TO_CACHE = [
  '/shadow',
  '/manifest.json',
  '/sw.js',
];

const API_ROUTES = [
  '/os/status',
  '/os/health',
  '/os/voice/tts',
  '/os/voice/voices',
  '/os/summon/abilities',
  '/os/summon/agent-templates',
  '/os/device',
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(ASSETS_TO_CACHE))
  );
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME && k !== API_CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

function getQueue() {
  try {
    return JSON.parse(self.sessionStorage.getItem(OFFLINE_QUEUE) || '[]');
  } catch {
    return [];
  }
}

function saveQueue(queue) {
  try {
    self.sessionStorage.setItem(OFFLINE_QUEUE, JSON.stringify(queue.slice(0, MAX_QUEUE_SIZE)));
  } catch {
    // sessionStorage may not be available in all SW contexts
  }
  return Promise.resolve();
}

function addToQueue(request) {
  return request.clone().text().then(body => {
    const entry = {
      url: request.url,
      method: request.method,
      headers: Object.fromEntries(request.headers.entries()),
      body: body,
      timestamp: Date.now(),
      retryCount: 0,
    };
    const queue = getQueue();
    if (queue.length >= MAX_QUEUE_SIZE) {
      queue.shift();
    }
    queue.push(entry);
    return saveQueue(queue);
  });
}

async function replayQueue() {
  const queue = getQueue();
  if (queue.length === 0) return;
  const failed = [];
  for (const entry of queue) {
    try {
      const headers = new Headers(entry.headers);
      const response = await fetch(entry.url, {
        method: entry.method,
        headers: headers,
        body: entry.method !== 'GET' ? entry.body : undefined,
      });
      if (!response.ok && response.status >= 500) {
        entry.retryCount = (entry.retryCount || 0) + 1;
        if (entry.retryCount < 3) {
          failed.push(entry);
        }
      }
    } catch {
      entry.retryCount = (entry.retryCount || 0) + 1;
      if (entry.retryCount < 3) {
        failed.push(entry);
      }
    }
  }
  saveQueue(failed);
  const clients = await self.clients.matchAll();
  clients.forEach(client => {
    client.postMessage({ type: 'QUEUE_REPLAYED', count: queue.length - failed.length });
  });
}

self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  const isPOST = event.request.method === 'POST';
  const isPUT = event.request.method === 'PUT';
  const isDELETE = event.request.method === 'DELETE';
  const isMutation = isPOST || isPUT || isDELETE;

  if (isMutation) {
    event.respondWith(
      fetch(event.request).catch(async () => {
        if (isPOST && url.pathname.startsWith('/os/')) {
          await addToQueue(event.request);
          const clients = await self.clients.matchAll();
          clients.forEach(client => {
            client.postMessage({ type: 'OFFLINE_QUEUED', url: url.pathname });
          });
          return new Response(JSON.stringify({ queued: true, offline: true }), {
            status: 202,
            headers: { 'Content-Type': 'application/json' },
          });
        }
        return new Response(JSON.stringify({ error: 'offline' }), {
          status: 503,
          headers: { 'Content-Type': 'application/json' },
        });
      })
    );
    return;
  }

  if (url.pathname.startsWith('/os/')) {
    event.respondWith(
      caches.open(API_CACHE_NAME).then(async cache => {
        const cached = await cache.match(event.request);
        if (cached) {
          const age = Date.now() - parseInt(cached.headers.get('sw-cache-time') || '0');
          if (age < API_CACHE_TTL) return cached;
        }
        try {
          const response = await fetch(event.request);
          if (response.ok) {
            const cloned = response.clone();
            const headers = new Headers(cloned.headers);
            headers.set('sw-cache-time', Date.now().toString());
            const body = await cloned.blob();
            const stamped = new Response(body, { status: cloned.status, statusText: cloned.statusText, headers });
            cache.put(event.request, stamped);
          }
          return response;
        } catch {
          if (cached) return cached;
          return new Response(JSON.stringify({ error: 'offline' }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' },
          });
        }
      })
    );
    return;
  }

  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) return cached;
      return fetch(event.request).then(response => {
        if (response.ok && (url.pathname === '/shadow' || url.pathname === '/manifest.json')) {
          const cloned = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, cloned));
        }
        return response;
      }).catch(() => {
        if (event.request.destination === 'document') {
          return caches.match('/shadow');
        }
        return new Response('Offline', { status: 503 });
      });
    })
  );
});

self.addEventListener('sync', event => {
  if (event.tag === 'shadowos-offline-queue') {
    event.waitUntil(replayQueue());
  }
});

self.addEventListener('push', event => {
  if (!event.data) return;
  try {
    const data = event.data.json();
    const title = data.title || 'ShadowOS';
    const options = {
      body: data.body || data.message || 'New notification',
      icon: data.icon || '/static/icon-192.png',
      badge: data.badge || '/static/icon-192.png',
      tag: data.tag || 'shadowos-' + Date.now(),
      data: data.data || {},
      vibrate: data.vibrate || [100, 50, 100],
      requireInteraction: data.requireInteraction || false,
      actions: data.actions || [],
    };
    event.waitUntil(self.registration.showNotification(title, options));
  } catch {
    event.waitUntil(
      self.registration.showNotification('ShadowOS', {
        body: 'You have a new notification',
        icon: '/static/icon-192.png',
        tag: 'shadowos-' + Date.now(),
      })
    );
  }
});

self.addEventListener('notificationclick', event => {
  event.notification.close();
  const targetUrl = event.notification.data?.url || '/shadow';
  event.waitUntil(
    self.clients.matchAll({ type: 'window' }).then(clients => {
      for (const client of clients) {
        if (client.url.includes('/shadow') && 'focus' in client) {
          return client.focus();
        }
      }
      return self.clients.openWindow(targetUrl);
    })
  );
});

self.addEventListener('message', event => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
  if (event.data && event.data.type === 'REPLAY_QUEUE') {
    event.waitUntil(replayQueue());
  }
});