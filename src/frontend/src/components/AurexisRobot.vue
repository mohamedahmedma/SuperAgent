<template>
  <div ref="host" class="aurexis-robot" aria-hidden="true"></div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue';
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

const host = ref<HTMLDivElement | null>(null);

let renderer: THREE.WebGLRenderer | null = null;
let frameId = 0;
let resizeObserver: ResizeObserver | null = null;
let scene: THREE.Scene | null = null;
let onPointerMove: ((event: PointerEvent) => void) | null = null;
let onPointerLeave: (() => void) | null = null;
let onPasswordPrivacy: ((event: Event) => void) | null = null;

const materials: THREE.Material[] = [];
const geometries: THREE.BufferGeometry[] = [];

const trackMaterial = <T extends THREE.Material>(material: T): T => {
  materials.push(material);
  return material;
};

const trackGeometry = <T extends THREE.BufferGeometry>(geometry: T): T => {
  geometries.push(geometry);
  return geometry;
};

const damp = (current: number, target: number, speed: number, delta: number) =>
  THREE.MathUtils.lerp(current, target, 1 - Math.exp(-speed * delta));

onMounted(() => {
  if (!host.value) return;

  scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
  camera.position.set(0, 0.05, 10.6);
  camera.lookAt(0, -0.45, 0);

  renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.7));
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1;
  host.value.appendChild(renderer.domElement);

  // Exact AUREXIS robot palette/material values.
  const ROBOT_SILVER = new THREE.Color('#B8C0CA');
  const ROBOT_SILVER_HIGHLIGHT = new THREE.Color('#D8DEE5');
  const ROBOT_FACE = new THREE.Color('#02060B');
  const ROBOT_NAVY = new THREE.Color('#07152E');
  const ROBOT_WHITE = new THREE.Color('#FFFFFF');

  const shell = trackMaterial(new THREE.MeshPhysicalMaterial({
    color: ROBOT_SILVER.clone(),
    metalness: 0.78,
    roughness: 0.27,
    clearcoat: 0.3,
    clearcoatRoughness: 0.2,
    reflectivity: 0.72,
  }));

  const shellSecondary = trackMaterial(new THREE.MeshPhysicalMaterial({
    color: ROBOT_SILVER_HIGHLIGHT.clone(),
    metalness: 0.82,
    roughness: 0.22,
    clearcoat: 0.28,
    clearcoatRoughness: 0.17,
    reflectivity: 0.76,
  }));

  const faceMaterial = trackMaterial(new THREE.MeshPhysicalMaterial({
    color: ROBOT_FACE.clone(),
    metalness: 0,
    roughness: 0.18,
    clearcoat: 0.5,
    clearcoatRoughness: 0.12,
    envMapIntensity: 0.12,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
  }));

  const darkMetal = trackMaterial(new THREE.MeshPhysicalMaterial({
    color: ROBOT_NAVY.clone(),
    metalness: 0.6,
    roughness: 0.3,
    clearcoat: 0.32,
    clearcoatRoughness: 0.19,
  }));

  const navyDetail = trackMaterial(new THREE.MeshStandardMaterial({
    color: ROBOT_NAVY.clone(),
    emissive: ROBOT_NAVY.clone(),
    emissiveIntensity: 0.015,
    metalness: 0.58,
    roughness: 0.32,
  }));

  const eyeMaterial = trackMaterial(new THREE.MeshBasicMaterial({
    color: ROBOT_WHITE.clone(),
    toneMapped: false,
    depthWrite: false,
    depthTest: false,
    side: THREE.DoubleSide,
  }));

  const eyeGlow = trackMaterial(new THREE.MeshBasicMaterial({
    color: ROBOT_WHITE.clone(),
    transparent: true,
    opacity: 0.13,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    depthTest: false,
    toneMapped: false,
    side: THREE.DoubleSide,
  }));

  const seam = trackMaterial(new THREE.MeshStandardMaterial({
    color: ROBOT_NAVY.clone(),
    metalness: 0.5,
    roughness: 0.34,
    transparent: true,
    opacity: 0.72,
    depthWrite: false,
  }));

  const reflection = trackMaterial(new THREE.MeshBasicMaterial({
    color: ROBOT_WHITE.clone(),
    transparent: true,
    opacity: 0.08,
    depthWrite: false,
  }));

  const antennaMaterial = trackMaterial(new THREE.MeshPhysicalMaterial({
    color: ROBOT_SILVER_HIGHLIGHT.clone(),
    metalness: 0.78,
    roughness: 0.24,
    transparent: true,
    opacity: 0.9,
    clearcoat: 0.26,
    clearcoatRoughness: 0.18,
  }));

  const robot = new THREE.Group();
  robot.scale.setScalar(0.78);
  robot.position.y = -0.05;
  scene.add(robot);

  // HEAD — exact source dimensions.
  const head = new THREE.Group();
  head.position.set(0, 0.84, 0);
  robot.add(head);

  const headShell = new THREE.Mesh(
    trackGeometry(new THREE.SphereGeometry(1, 64, 48)),
    shell
  );
  headShell.scale.set(1.36, 1.23, 1);
  head.add(headShell);

  const jawShell = new THREE.Mesh(
    trackGeometry(new THREE.SphereGeometry(1, 56, 40)),
    shellSecondary
  );
  jawShell.position.set(0, -0.58, 0.03);
  jawShell.scale.set(1.03, 0.65, 0.87);
  head.add(jawShell);

  const seamCurves = [
    new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 1.12, 0.5),
      new THREE.Vector3(0, 0.83, 0.82),
      new THREE.Vector3(0, 0.55, 0.94),
    ]),
    new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.78, 0.96, 0.73),
      new THREE.Vector3(-0.66, 0.48, 0.95),
      new THREE.Vector3(-0.77, -0.68, 0.81),
    ]),
    new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.78, 0.96, 0.73),
      new THREE.Vector3(0.66, 0.48, 0.95),
      new THREE.Vector3(0.77, -0.68, 0.81),
    ]),
  ];

  seamCurves.forEach((curve) => {
    head.add(new THREE.Mesh(trackGeometry(new THREE.TubeGeometry(curve, 32, 0.014, 8, false)), seam));
  });

  for (const side of [-1, 1] as const) {
    const fastener = new THREE.Mesh(trackGeometry(new THREE.SphereGeometry(1, 18, 14)), darkMetal);
    fastener.position.set(side * 0.96, 0.2, 0.91);
    fastener.scale.set(0.045, 0.045, 0.026);
    head.add(fastener);
  }

  const faceShape = new THREE.Shape();
  faceShape.moveTo(-1.04, 0.38);
  faceShape.lineTo(1.04, 0.38);
  faceShape.lineTo(0.86, -0.44);
  faceShape.lineTo(0, -0.68);
  faceShape.lineTo(-0.86, -0.44);
  faceShape.closePath();

  const face = new THREE.Mesh(trackGeometry(new THREE.ShapeGeometry(faceShape, 30)), faceMaterial);
  face.position.set(0, -0.04, 1.018);
  face.renderOrder = 2;
  head.add(face);

  // Subtle face-glass reflection matching the AUREXIS glossy mask presentation.
  const faceGlass = new THREE.Mesh(
    trackGeometry(new THREE.ShapeGeometry(faceShape, 30)),
    trackMaterial(new THREE.MeshBasicMaterial({
      color: new THREE.Color('#FFFFFF'),
      transparent: true,
      opacity: 0.025,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
      toneMapped: false,
    }))
  );
  faceGlass.position.set(0, -0.035, 1.024);
  faceGlass.renderOrder = 3;
  head.add(faceGlass);

  // Website-accurate eye root: both eyes move/blink as one group.
  const eyesRoot = new THREE.Group();
  head.add(eyesRoot);

  const eyeShape = new THREE.Shape();
  eyeShape.moveTo(-0.34, 0.11);
  eyeShape.quadraticCurveTo(-0.05, 0.2, 0.31, 0.08);
  eyeShape.lineTo(0.2, -0.17);
  eyeShape.quadraticCurveTo(-0.06, -0.22, -0.31, -0.11);
  eyeShape.closePath();

  for (const side of [-1, 1] as const) {
    const eyeGroup = new THREE.Group();
    eyeGroup.position.set(side * 0.43, -0.08, 1.075);
    eyeGroup.scale.set(side === 1 ? -1 : 1, 1, 1);

    const glow = new THREE.Mesh(trackGeometry(new THREE.ShapeGeometry(eyeShape, 30)), eyeGlow);
    glow.scale.set(1.12, 1.18, 1);
    glow.renderOrder = 10;
    eyeGroup.add(glow);

    const eye = new THREE.Mesh(trackGeometry(new THREE.ShapeGeometry(eyeShape, 30)), eyeMaterial);
    eye.position.z = 0.012;
    eye.scale.set(0.9, 0.88, 1);
    eye.renderOrder = 11;
    eyeGroup.add(eye);

    eyesRoot.add(eyeGroup);
  }

  const foreheadBar = new THREE.Mesh(
    trackGeometry(new RoundedBoxGeometry(0.66, 0.05, 0.2, 6, 0.025)),
    navyDetail
  );
  foreheadBar.position.set(0, 1.215, 0.08);
  head.add(foreheadBar);

  const headReflection = new THREE.Mesh(trackGeometry(new THREE.SphereGeometry(1, 24, 16)), reflection);
  headReflection.position.set(-0.72, 0.82, 0.82);
  headReflection.scale.set(0.24, 0.08, 0.025);
  headReflection.rotation.z = -0.35;
  head.add(headReflection);

  // ANTENNAE — exact source dimensions.
  for (const side of [-1, 1] as const) {
    const antenna = new THREE.Group();
    antenna.position.set(side * 1.39, 0.02, -0.02);
    head.add(antenna);

    const baseMesh = new THREE.Mesh(trackGeometry(new THREE.CylinderGeometry(0.16, 0.16, 0.2, 28)), shellSecondary);
    baseMesh.rotation.z = Math.PI / 2;
    antenna.add(baseMesh);

    const rod = new THREE.Mesh(trackGeometry(new THREE.CylinderGeometry(0.035, 0.045, 2.5, 20)), antennaMaterial);
    rod.position.set(side * 0.045, 1.25, 0);
    antenna.add(rod);

    [2.1, 2.25, 2.4].forEach((height, index) => {
      const tipGroup = new THREE.Group();
      tipGroup.position.set(side * 0.045, height, 0);
      antenna.add(tipGroup);

      const ball = new THREE.Mesh(
        trackGeometry(new THREE.SphereGeometry(0.055 - index * 0.004, 20, 16)),
        navyDetail
      );
      tipGroup.add(ball);

      const ring = new THREE.Mesh(trackGeometry(new THREE.TorusGeometry(0.065, 0.008, 8, 24)), shellSecondary);
      ring.position.z = -0.002;
      ring.rotation.x = Math.PI / 2;
      tipGroup.add(ring);
    });
  }

  // BODY — exact source profile and position.
  const profile = [
    new THREE.Vector2(0.42, 1.08),
    new THREE.Vector2(0.72, 1.04),
    new THREE.Vector2(0.94, 0.88),
    new THREE.Vector2(1.02, 0.5),
    new THREE.Vector2(0.97, 0.08),
    new THREE.Vector2(0.82, -0.42),
    new THREE.Vector2(0.58, -0.82),
    new THREE.Vector2(0.32, -1.05),
    new THREE.Vector2(0.12, -1.13),
  ];

  const bodyGroup = new THREE.Group();
  bodyGroup.position.set(0, -1.72, 0);
  robot.add(bodyGroup);

  const body = new THREE.Mesh(trackGeometry(new THREE.LatheGeometry(profile, 64)), shell);
  body.scale.set(1, 1, 0.78);
  bodyGroup.add(body);

  const lowerSeamPoints = Array.from({ length: 41 }, (_, index) => {
    const angle = (index / 40) * Math.PI * 2;
    return new THREE.Vector3(Math.cos(angle) * 0.66, -0.62, Math.sin(angle) * 0.5);
  });
  const lowerSeam = new THREE.CatmullRomCurve3(lowerSeamPoints, true);
  bodyGroup.add(new THREE.Mesh(trackGeometry(new THREE.TubeGeometry(lowerSeam, 64, 0.015, 8, true)), seam));

  const bodyReflection = new THREE.Mesh(trackGeometry(new THREE.SphereGeometry(1, 24, 18)), reflection);
  bodyReflection.position.set(-0.35, 0.58, 0.72);
  bodyReflection.scale.set(0.28, 0.09, 0.035);
  bodyReflection.rotation.z = -0.28;
  bodyGroup.add(bodyReflection);

  // ARMS — exact AUREXIS arm dimensions.
  const LEFT = -1 as const;
  const RIGHT = 1 as const;
  const shoulderAnchor = { x: 1.14, y: -1.32, z: -0.12 };
  const leftShoulderRest = -0.08;
  const rightShoulderRest = 0.08;

  const makeArm = (side: -1 | 1) => {
    const shoulder = new THREE.Group();
    shoulder.position.set(side * shoulderAnchor.x, shoulderAnchor.y, shoulderAnchor.z);
    shoulder.rotation.z = side === -1 ? leftShoulderRest : rightShoulderRest;
    robot.add(shoulder);

    const pivot = new THREE.Mesh(trackGeometry(new THREE.SphereGeometry(1, 36, 28)), darkMetal);
    pivot.position.set(side * -0.015, 0, 0);
    pivot.scale.set(0.27, 0.27, 0.25);
    shoulder.add(pivot);

    const upper = new THREE.Mesh(
      trackGeometry(new RoundedBoxGeometry(0.44, 0.6, 0.4, 8, 0.19)),
      shell
    );
    upper.position.set(side * 0.055, -0.29, 0);
    upper.rotation.z = side * 0.035;
    shoulder.add(upper);

    const upperReflection = new THREE.Mesh(trackGeometry(new THREE.SphereGeometry(1, 18, 14)), reflection);
    upperReflection.position.set(side * -0.055, -0.16, 0.19);
    upperReflection.scale.set(0.1, 0.19, 0.025);
    upperReflection.rotation.z = side * -0.08;
    shoulder.add(upperReflection);

    const forearm = new THREE.Group();
    forearm.position.set(side * 0.07, -0.54, 0);
    shoulder.add(forearm);

    const wrist = new THREE.Mesh(trackGeometry(new THREE.TorusGeometry(0.175, 0.022, 10, 36)), darkMetal);
    wrist.position.y = -0.02;
    wrist.rotation.x = Math.PI / 2;
    forearm.add(wrist);

    const lower = new THREE.Mesh(
      trackGeometry(new RoundedBoxGeometry(0.34, 0.43, 0.32, 8, 0.15)),
      shellSecondary
    );
    lower.position.set(0, -0.22, 0);
    forearm.add(lower);

    const hand = new THREE.Mesh(trackGeometry(new THREE.SphereGeometry(1, 36, 28)), shell);
    hand.position.set(0, -0.47, 0);
    hand.scale.set(0.22, 0.21, 0.22);
    forearm.add(hand);

    const handDetail = new THREE.Mesh(trackGeometry(new THREE.SphereGeometry(1, 18, 14)), darkMetal);
    handDetail.position.set(0, -0.425, 0.195);
    handDetail.scale.set(0.12, 0.035, 0.018);
    forearm.add(handDetail);

    return { shoulder, forearm };
  };

  const leftArm = makeArm(LEFT);
  makeArm(RIGHT);

  // Ground shadow/glow intentionally removed in V10.2.

  // Exact AUREXIS lighting values.
  scene.add(new THREE.AmbientLight('#D7DEE7', 0.58));

  const key = new THREE.DirectionalLight('#FFFFFF', 2.2);
  key.position.set(-3.6, 5.4, 5.8);
  scene.add(key);

  const fill = new THREE.DirectionalLight('#C7CFD9', 0.72);
  fill.position.set(4.2, 1.5, 4.4);
  scene.add(fill);

  const rear = new THREE.PointLight('#EDF2F7', 1.85, 8);
  rear.position.set(3.4, 1.8, -2.4);
  scene.add(rear);

  const front = new THREE.PointLight('#FFFFFF', 0.45, 7);
  front.position.set(-2.8, -1.2, 3.8);
  scene.add(front);

  // Exact AUREXIS website pointer model:
  // normalize pointer against the ROBOT CENTER, not against the robot canvas bounds.
  const cursor = {
    x: 0,
    y: 0,
    active: false,
  };

  // Password privacy choreography:
  // idle = normal mouse tracking
  // away = password hidden, robot deliberately looks away
  // peek = password revealed, head stays mostly away while eyes sneak a glance back
  let passwordPrivacyMode: 'idle' | 'away' | 'peek' = 'idle';

  onPasswordPrivacy = (event: Event) => {
    const customEvent = event as CustomEvent<{ mode?: 'idle' | 'away' | 'peek' }>;
    const nextMode = customEvent.detail?.mode;

    if (nextMode === 'away' || nextMode === 'peek' || nextMode === 'idle') {
      passwordPrivacyMode = nextMode;
    }
  };

  if (onPasswordPrivacy) {
    window.addEventListener('aurexis-password-privacy', onPasswordPrivacy as EventListener);
  }

  onPointerMove = (event: PointerEvent) => {
    if (event.pointerType === 'touch' || !host.value) return;

    const rect = host.value.getBoundingClientRect();
    const robotX = rect.left + rect.width / 2;
    const robotY = rect.top + rect.height / 2;

    const horizontalRange = Math.max(170, window.innerWidth * 0.16);
    const verticalRange = Math.max(150, window.innerHeight * 0.20);

    cursor.x = THREE.MathUtils.clamp(
      (event.clientX - robotX) / horizontalRange,
      -1,
      1
    );

    cursor.y = THREE.MathUtils.clamp(
      (robotY - event.clientY) / verticalRange,
      -1,
      1
    );

    cursor.active = true;
  };

  onPointerLeave = () => {
    cursor.x = 0;
    cursor.y = 0;
    cursor.active = false;
  };

  const coarsePointer = window.matchMedia(
    '(hover: none), (pointer: coarse)'
  ).matches;

  if (!coarsePointer && onPointerMove) {
    window.addEventListener('pointermove', onPointerMove, { passive: true });
  }
  if (onPointerLeave) {
    window.addEventListener('blur', onPointerLeave);
    document.documentElement.addEventListener('mouseleave', onPointerLeave);
  }

  const resize = () => {
    if (!host.value || !renderer) return;
    const width = Math.max(1, host.value.clientWidth);
    const height = Math.max(1, host.value.clientHeight);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };

  resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(host.value);
  resize();

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let previous = performance.now();
  const startedAt = performance.now();

  // Website-accurate blink schedule: 3.1–7.0 seconds between blinks.
  let nextBlinkAt = startedAt / 1000 + 3.1 + Math.random() * 3.9;
  let blinkStartedAt = -1;

  const animate = (now: number) => {
    if (!renderer || !scene) return;

    const delta = Math.min(0.05, (now - previous) / 1000);
    previous = now;

    // AUREXIS greeting cycle:
    // 2.6s waving -> arm lowers naturally -> 5s rest -> repeat.
    if (!reduceMotion) {
      const elapsed = Math.max(0, (now - startedAt) / 1000);

      const WAVE_SECONDS = 2.6;
      const REST_SECONDS = 5.0;
      const CYCLE_SECONDS = WAVE_SECONDS + REST_SECONDS;
      const phase = elapsed % CYCLE_SECONDS;

      let leftShoulderTarget = leftShoulderRest;
      let leftForearmTarget = 0;

      if (phase < WAVE_SECONDS) {
        // Smoothly raise the arm at the beginning and lower it near the end.
        const raiseIn = THREE.MathUtils.smoothstep(phase, 0.0, 0.42);
        const lowerOut = 1 - THREE.MathUtils.smoothstep(
          phase,
          WAVE_SECONDS - 0.48,
          WAVE_SECONDS
        );
        const waveAmount = Math.min(raiseIn, lowerOut);

        // Friendly wave: about 4 full hand waves during the greeting window.
        const waveOscillation = Math.sin(phase * Math.PI * 3.05);

        const shoulderWaveOffset = (1.27 + waveOscillation * 0.07) * waveAmount;
        const forearmWaveOffset = (0.10 + waveOscillation * 0.30) * waveAmount;

        leftShoulderTarget = leftShoulderRest - shoulderWaveOffset;
        leftForearmTarget = -forearmWaveOffset;
      }

      // During the 5-second rest both targets are the natural resting pose.
      leftArm.shoulder.rotation.z = damp(
        leftArm.shoulder.rotation.z,
        leftShoulderTarget,
        phase < WAVE_SECONDS ? 7.0 : 5.2,
        delta
      );

      leftArm.forearm.rotation.z = damp(
        leftArm.forearm.rotation.z,
        leftForearmTarget,
        phase < WAVE_SECONDS ? 7.4 : 5.6,
        delta
      );
    }

    if (!reduceMotion) {
      const t = now / 1000;
      const pointerTracking = cursor.active;

      // EXACT AUREXIS site head tracking values.
      const HEAD_YAW_LIMIT = THREE.MathUtils.degToRad(34);
      const HEAD_PITCH_LIMIT = THREE.MathUtils.degToRad(19);
      const HEAD_TRACKING_DAMPING = 6.5;

      let targetYaw = 0;
      let targetPitch = 0;
      let targetRoll = 0;

      if (passwordPrivacyMode === 'away') {
        // Respectful privacy pose: turn outward and slightly down.
        targetYaw = THREE.MathUtils.degToRad(27);
        targetPitch = THREE.MathUtils.degToRad(4);
        targetRoll = THREE.MathUtils.degToRad(-2);
      } else if (passwordPrivacyMode === 'peek') {
        // "Sneaky peek": head still turned away, eyes glance back toward the password field.
        targetYaw = THREE.MathUtils.degToRad(18);
        targetPitch = THREE.MathUtils.degToRad(-3);
        targetRoll = THREE.MathUtils.degToRad(-4);
      } else if (pointerTracking) {
        targetYaw = THREE.MathUtils.clamp(
          cursor.x * HEAD_YAW_LIMIT,
          -HEAD_YAW_LIMIT,
          HEAD_YAW_LIMIT
        );
        targetPitch = THREE.MathUtils.clamp(
          -cursor.y * HEAD_PITCH_LIMIT,
          -HEAD_PITCH_LIMIT,
          HEAD_PITCH_LIMIT
        );
      }

      head.rotation.y = damp(
        head.rotation.y,
        targetYaw,
        HEAD_TRACKING_DAMPING,
        delta
      );
      head.rotation.x = damp(
        head.rotation.x,
        targetPitch,
        HEAD_TRACKING_DAMPING,
        delta
      );
      head.rotation.z = damp(
        head.rotation.z,
        targetRoll,
        5.2,
        delta
      );

      // EXACT AUREXIS site blink timing/profile.
      if (t >= nextBlinkAt) {
        blinkStartedAt = t;
        nextBlinkAt = t + 3.1 + Math.random() * 3.9;
      }

      const blinkElapsed = t - blinkStartedAt;
      const blinkActive =
        blinkStartedAt >= 0 && blinkElapsed < 0.18;

      const blink = blinkActive
        ? blinkElapsed < 0.065
          ? THREE.MathUtils.lerp(1, 0.08, blinkElapsed / 0.065)
          : blinkElapsed < 0.09
            ? 0.08
            : THREE.MathUtils.lerp(
                0.08,
                1,
                (blinkElapsed - 0.09) / 0.09
              )
        : 1;

      if (!blinkActive) {
        blinkStartedAt = -1;
        eyesRoot.scale.y = 1;
      }

      // EXACT AUREXIS site eye tracking range and damping.
      let eyeTrackX = 0;
      let eyeTrackY = 0;

      if (passwordPrivacyMode === 'away') {
        // Eyes reinforce the "not looking" pose.
        eyeTrackX = 0.036;
        eyeTrackY = -0.010;
      } else if (passwordPrivacyMode === 'peek') {
        // Head stays turned out, eyes subtly glance back toward the form.
        eyeTrackX = -0.044;
        eyeTrackY = 0.014;
      } else if (pointerTracking) {
        eyeTrackX = THREE.MathUtils.clamp(cursor.x * 0.046, -0.046, 0.046);
        eyeTrackY = THREE.MathUtils.clamp(cursor.y * 0.028, -0.028, 0.028);
      }

      if (blinkActive) {
        eyesRoot.scale.y = blink;
      }

      eyesRoot.position.x = damp(
        eyesRoot.position.x,
        eyeTrackX,
        8,
        delta
      );
      eyesRoot.position.y = damp(
        eyesRoot.position.y,
        eyeTrackY,
        8,
        delta
      );
    }

    renderer.render(scene, camera);
    frameId = requestAnimationFrame(animate);
  };

  frameId = requestAnimationFrame(animate);
});

onBeforeUnmount(() => {
  cancelAnimationFrame(frameId);
  resizeObserver?.disconnect();
  if (onPointerMove) {
    window.removeEventListener('pointermove', onPointerMove);
    onPointerMove = null;
  }
  if (onPointerLeave) {
    window.removeEventListener('blur', onPointerLeave);
    document.documentElement.removeEventListener('mouseleave', onPointerLeave);
    onPointerLeave = null;
  }
  if (onPasswordPrivacy) {
    window.removeEventListener('aurexis-password-privacy', onPasswordPrivacy as EventListener);
    onPasswordPrivacy = null;
  }

  materials.forEach((material) => material.dispose());
  geometries.forEach((geometry) => geometry.dispose());

  if (renderer) {
    renderer.dispose();
    renderer.domElement.remove();
  }

  renderer = null;
  scene = null;
});
</script>
