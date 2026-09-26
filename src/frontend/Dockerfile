# syntax=docker/dockerfile:1.7
FROM node:24.19.0-alpine AS builder
WORKDIR /app
ARG VITE_IDENTITY_BASE_URL=""
ENV VITE_IDENTITY_BASE_URL=${VITE_IDENTITY_BASE_URL}
COPY src/frontend/package.json src/frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY src/frontend/ ./
RUN npm run build

FROM nginx:1.27.3-alpine
COPY src/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1