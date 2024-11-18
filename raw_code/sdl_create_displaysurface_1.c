static DisplaySurface* sdl_create_displaysurface(int width, int height)

{

    DisplaySurface *surface = (DisplaySurface*) g_malloc0(sizeof(DisplaySurface));

    if (surface == NULL) {

        fprintf(stderr, "sdl_create_displaysurface: malloc failed\n");

        exit(1);

    }



    surface->width = width;

    surface->height = height;



    if (scaling_active) {

        int linesize;

        PixelFormat pf;

        if (host_format.BytesPerPixel != 2 && host_format.BytesPerPixel != 4) {

            linesize = width * 4;

            pf = qemu_default_pixelformat(32);

        } else {

            linesize = width * host_format.BytesPerPixel;

            pf = sdl_to_qemu_pixelformat(&host_format);

        }

        qemu_alloc_display(surface, width, height, linesize, pf, 0);

        return surface;

    }



    if (host_format.BitsPerPixel == 16)

        do_sdl_resize(width, height, 16);

    else

        do_sdl_resize(width, height, 32);



    surface->pf = sdl_to_qemu_pixelformat(real_screen->format);

    surface->linesize = real_screen->pitch;

    surface->data = real_screen->pixels;



#ifdef HOST_WORDS_BIGENDIAN

    surface->flags = QEMU_REALPIXELS_FLAG | QEMU_BIG_ENDIAN_FLAG;

#else

    surface->flags = QEMU_REALPIXELS_FLAG;

#endif

    allocator = 1;



    return surface;

}
