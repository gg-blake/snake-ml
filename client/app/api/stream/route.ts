import type { NextApiRequest, NextApiResponse } from 'next';
import NextCors from 'nextjs-cors';
import http from 'http'; // Make sure http is imported

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  // Handle CORS for localhost:8000
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle preflight OPTIONS request
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  // Restrict to only GET requests
  if (req.method !== 'GET') {
    res.status(405).json({ message: 'Method Not Allowed' });
    return;
  }

  // Set the response headers for Server-Sent Events (SSE)
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  // Connect to the external EventSource stream
  const externalRequest = http.request(
    {
      hostname: 'localhost',
      port: 8000,
      path: '/stream', // This is the path from the external server
      method: 'GET',
    },
    (externalResponse) => {
      externalResponse.on('data', (chunk) => {
        // Forward the event chunk as SSE (Server-Sent Events) format
        res.write(`data: ${chunk.toString()}\n\n`);
      });

      externalResponse.on('end', () => {
        res.end();
      });

      externalResponse.on('error', (err) => {
        console.error('Error in external stream:', err);
        res.end();
      });
    }
  );

  externalRequest.on('error', (err) => {
    console.error('Error connecting to external stream:', err);
    res.status(500).end('Error connecting to external stream.');
  });

  externalRequest.end();

  // Close connection if client closes
  req.on('close', () => {
    console.log('Client closed connection');
    externalRequest.destroy();
    res.end();
  });
}
