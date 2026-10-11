from pathlib import Path
import math
import sys
import numpy as np
import skia
from flowphi.charts.blocks import compose


ASSETS = Path(__file__).with_name('block_assets')


def noise_texture(size, seed, sparse=False):
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
    if sparse:
        values = np.where(values > 239, values, 0).astype(np.uint8)
    rgba = np.empty((size, size, 4), dtype=np.uint8)
    rgba[:, :, :3] = values[:, :, None]
    rgba[:, :, 3] = 255
    return skia.Image.fromarray(rgba, colorType=skia.ColorType.kRGBA_8888_ColorType)


def _path(geometry):
    path = skia.Path()
    path.setFillType(skia.PathFillType.kEvenOdd)
    polygons = [geometry] if geometry.geom_type == 'Polygon' else getattr(geometry, 'geoms', ())
    for polygon in polygons:
        if polygon.geom_type != 'Polygon':
            continue
        for ring in (polygon.exterior, *polygon.interiors):
            xy = list(ring.coords)
            if not xy: continue
            path.moveTo(*xy[0])
            for p in xy[1:]: path.lineTo(*p)
            path.close()
    return path


def _dilate(a):
    padded = np.pad(a, 1, mode='edge')
    h, w = a.shape
    return np.maximum.reduce([padded[y:y + h, x:x + w] for y in range(3) for x in range(3)])


class BlockRenderer:
    def __init__(self, blocks, width, height, use_gpu=True, boundary_mode='native'):
        if boundary_mode not in ('aligned', 'native'):
            raise ValueError(f'Unknown block boundary mode: {boundary_mode}')
        self.boundary_mode = boundary_mode
        self.blocks, self.width, self.height = blocks, width, height
        self.native = self.gpu = None
        if use_gpu:
            try:
                import glcontext
                backend = glcontext.get_backend_by_name('egl') if sys.platform.startswith('linux') else glcontext.default_backend()
                self.native = backend(glversion=330, mode='standalone')
                with self.native:
                    self.gpu = skia.GrDirectContext.MakeGL()
                if self.gpu is None:
                    self.native.release()
                    self.native = None
            except Exception as exc:
                self.native = self.gpu = None
                print(f'BLOCK_RENDER raster fallback: {type(exc).__name__}: {exc}', flush=True)
        print(f'BLOCK_RENDER backend={"OpenGL" if self.gpu else "raster"}', flush=True)
        self.size = (max(1, width // 8), max(1, height // 8))
        self.effect_size = tuple(n * 2 for n in self.size)
        self.scene_size = (max(1, width // 6), max(1, height // 6))
        self.effects = {name: skia.RuntimeEffect.MakeForShader((ASSETS / f'{name}.sksl').read_text())
                      for name in ('compose', 'active', 'disabled')}
        self.noise = noise_texture(256, 2026).makeShader(
            skia.TileMode.kMirror, skia.TileMode.kMirror, skia.SamplingOptions(skia.FilterMode.kNearest))
        self.spark = noise_texture(128, 104, sparse=True).makeShader(
            skia.TileMode.kRepeat, skia.TileMode.kRepeat, skia.SamplingOptions(skia.FilterMode.kNearest))
        self.black = skia.Shaders.Color(skia.ColorBLACK)

    def surface(self, size):
        if self.gpu is None:
            return skia.Surface(*size)
        surface = skia.Surface.MakeRenderTarget(self.gpu, skia.Budgeted.kNo, skia.ImageInfo.MakeN32Premul(*size))
        if surface is None:
            raise RuntimeError('Cannot allocate the block effect surface')
        return surface

    def mask(self, geometry):
        s = self.surface(self.size)
        c = s.getCanvas(); c.clear(skia.ColorBLACK)
        c.scale(self.size[0] / self.blocks.width, self.size[1] / self.blocks.height)
        c.drawPath(_path(geometry), skia.Paint(Color=skia.ColorWHITE, AntiAlias=False))
        return s.makeImageSnapshot()

    @staticmethod
    def sampler(image, linear=False):
        return image.makeShader(skia.TileMode.kClamp, skia.TileMode.kClamp,
            skia.SamplingOptions(skia.FilterMode.kLinear if linear else skia.FilterMode.kNearest))

    def shader(self, name, seconds, children, **uniforms):
        b = skia.RuntimeShaderBuilder(self.effects[name])
        b.setUniform('resolution', (float(self.width), float(self.height)))
        b.setUniform('time', float(seconds))
        for key, value in uniforms.items():
            b.setUniform(key, tuple(float(n) for n in value) if isinstance(value, tuple) else value)
        for key, value in children.items(): b.setChild(key, value)
        return b.makeShader()

    def active_mask(self, geometry, seconds):
        raw = self.mask(geometry)
        target = self.surface(self.size)
        shader = self.shader('compose', seconds,
            {'mask': self.sampler(raw), 'noise': self.noise},
            resolution=self.size,
            displacementStrength=.1 if self.boundary_mode == 'native' else 0.)
        target.getCanvas().drawPaint(skia.Paint(Shader=shader))
        return target.makeImageSnapshot()

    def draw(self, canvas, seconds):
        if self.gpu is None:
            return self._draw(canvas, seconds)
        if not any(b.phase(seconds) != 'hidden' for b in self.blocks.areas):
            return
        # Upload the already-rendered chart once, do only the block passes on
        # this private context, then return a raster frame to the existing encoder.
        with self.native:
            target = self.surface((self.width, self.height))
            target.getCanvas().drawImage(canvas.getSurface().makeImageSnapshot(), 0, 0)
            self._draw(target.getCanvas(), seconds)
            result = target.makeImageSnapshot().makeRasterImage()
        canvas.drawImage(result, 0, 0, paint=skia.Paint(BlendMode=skia.BlendMode.kSrc))

    def _draw(self, canvas, seconds):
        phases = {'active': [], 'ready': [], 'disabled': []}
        for b in self.blocks.areas:
            phase = b.phase(seconds)
            if phase in phases: phases[phase].append(b.rectangle(seconds))
        if not any(phases.values()): return
        geometries = {k: compose(v) for k, v in phases.items()}
        # The APK uses 1/8-size point-filtered masks, 1/4-size effects and a
        # 1/6-size scene-color capture. Keep that characteristic blocky edge.
        mask = self.active_mask(geometries['active'], seconds)
        pixels = mask.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)[:, :, 0].astype(np.float32) / 255
        pixels = np.repeat(np.repeat(pixels, 2, axis=0), 2, axis=1)
        edge = np.clip(_dilate(pixels) - pixels, 0, 1)
        glow = np.zeros_like(pixels); grown = pixels
        denom = sum(n**2.65 for n in range(1, 7))
        for k in range(6):
            weight = (6 - k)**2.65 / denom
            if weight < .01: break
            expanded = _dilate(grown)
            glow += np.clip(expanded - grown, 0, 1) * (1 - pixels) * weight
            # Native ping-pong targets are RG16 (two UNorm8 channels).
            glow = np.rint(np.clip(glow, 0, 1) * 255) / 255
            grown = expanded
        rgba = np.zeros((*pixels.shape, 4), dtype=np.uint8)
        rgba[:, :, 0] = np.rint(edge * 255).astype(np.uint8)
        rgba[:, :, 1] = np.rint(np.clip(glow, 0, 1) * 255).astype(np.uint8)
        rgba[:, :, 3] = 255
        effect = skia.Image.fromarray(rgba, colorType=skia.ColorType.kRGBA_8888_ColorType)
        disabled = self.mask(geometries['disabled'])
        ready = self.mask(geometries['ready'])
        surface = canvas.getSurface()
        if surface is None: raise ValueError('Block effects require a raster surface')
        scene_surface = self.surface(self.scene_size)
        scene_surface.getCanvas().drawImageRect(surface.makeImageSnapshot(), skia.Rect.MakeWH(*self.scene_size),
            skia.SamplingOptions(skia.FilterMode.kLinear))
        children = {'mask': self.sampler(mask), 'effects': self.sampler(effect, True),
                  'ready': self.sampler(ready), 'scene': self.sampler(scene_surface.makeImageSnapshot()),
                  'noise': self.noise, 'spark': self.spark}
        active = self.shader('active', seconds, children, maskSize=self.size, effectSize=self.effect_size, sceneSize=self.scene_size)
        canvas.drawPaint(skia.Paint(Shader=active))
        if not geometries['disabled'].is_empty:
            shader = self.shader('disabled', seconds, {'mask': self.sampler(disabled), 'noise': self.noise, 'spark': self.spark}, maskSize=self.size)
            canvas.drawPaint(skia.Paint(Shader=shader, BlendMode=skia.BlendMode.kPlus))
