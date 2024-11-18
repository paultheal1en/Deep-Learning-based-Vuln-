static void load_cursor(VmncContext *c, const uint8_t *src)

{

    int i, j, p;

    const int bpp   = c->bpp2;

    uint8_t *dst8   =             c->curbits;

    uint16_t *dst16 = (uint16_t *)c->curbits;

    uint32_t *dst32 = (uint32_t *)c->curbits;



    for (j = 0; j < c->cur_h; j++) {

        for (i = 0; i < c->cur_w; i++) {

            p = vmnc_get_pixel(src, bpp, c->bigendian);

            src += bpp;

            if (bpp == 1)

                *dst8++ = p;

            if (bpp == 2)

                *dst16++ = p;

            if (bpp == 4)

                *dst32++ = p;

        }

    }

    dst8  =            c->curmask;

    dst16 = (uint16_t*)c->curmask;

    dst32 = (uint32_t*)c->curmask;

    for (j = 0; j < c->cur_h; j++) {

        for (i = 0; i < c->cur_w; i++) {

            p = vmnc_get_pixel(src, bpp, c->bigendian);

            src += bpp;

            if (bpp == 1)

                *dst8++ = p;

            if (bpp == 2)

                *dst16++ = p;

            if (bpp == 4)

                *dst32++ = p;

        }

    }

}
