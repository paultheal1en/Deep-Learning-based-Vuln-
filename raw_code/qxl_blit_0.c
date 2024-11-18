static void qxl_blit(PCIQXLDevice *qxl, QXLRect *rect)

{

    DisplaySurface *surface = qemu_console_surface(qxl->vga.con);

    uint8_t *dst = surface_data(surface);

    uint8_t *src;

    int len, i;



    if (is_buffer_shared(surface)) {

        return;

    }

    if (!qxl->guest_primary.data) {

        trace_qxl_render_blit_guest_primary_initialized();

        qxl->guest_primary.data = memory_region_get_ram_ptr(&qxl->vga.vram);

    }

    trace_qxl_render_blit(qxl->guest_primary.qxl_stride,

            rect->left, rect->right, rect->top, rect->bottom);

    src = qxl->guest_primary.data;

    if (qxl->guest_primary.qxl_stride < 0) {

        /* qxl surface is upside down, walk src scanlines

         * in reverse order to flip it */

        src += (qxl->guest_primary.surface.height - rect->top - 1) *

            qxl->guest_primary.abs_stride;

    } else {

        src += rect->top * qxl->guest_primary.abs_stride;

    }

    dst += rect->top  * qxl->guest_primary.abs_stride;

    src += rect->left * qxl->guest_primary.bytes_pp;

    dst += rect->left * qxl->guest_primary.bytes_pp;

    len  = (rect->right - rect->left) * qxl->guest_primary.bytes_pp;



    for (i = rect->top; i < rect->bottom; i++) {

        memcpy(dst, src, len);

        dst += qxl->guest_primary.abs_stride;

        src += qxl->guest_primary.qxl_stride;

    }

}
