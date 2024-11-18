void qxl_render_resize(PCIQXLDevice *qxl)

{

    QXLSurfaceCreate *sc = &qxl->guest_primary.surface;



    qxl->guest_primary.stride = sc->stride;

    qxl->guest_primary.resized++;

    switch (sc->format) {

    case SPICE_SURFACE_FMT_16_555:

        qxl->guest_primary.bytes_pp = 2;

        qxl->guest_primary.bits_pp = 15;

        break;

    case SPICE_SURFACE_FMT_16_565:

        qxl->guest_primary.bytes_pp = 2;

        qxl->guest_primary.bits_pp = 16;

        break;

    case SPICE_SURFACE_FMT_32_xRGB:

    case SPICE_SURFACE_FMT_32_ARGB:

        qxl->guest_primary.bytes_pp = 4;

        qxl->guest_primary.bits_pp = 32;

        break;

    default:

        fprintf(stderr, "%s: unhandled format: %x\n", __FUNCTION__,

                qxl->guest_primary.surface.format);

        qxl->guest_primary.bytes_pp = 4;

        qxl->guest_primary.bits_pp = 32;

        break;

    }

}
