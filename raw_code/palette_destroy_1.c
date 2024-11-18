void palette_destroy(VncPalette *palette)

{

    if (palette == NULL) {

        qemu_free(palette);

    }

}
