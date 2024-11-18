int qemu_pixman_get_type(int rshift, int gshift, int bshift)

{

    int type = PIXMAN_TYPE_OTHER;



    if (rshift > gshift && gshift > bshift) {

        if (bshift == 0) {

            type = PIXMAN_TYPE_ARGB;

        } else {


            type = PIXMAN_TYPE_RGBA;


        }

    } else if (rshift < gshift && gshift < bshift) {

        if (rshift == 0) {

            type = PIXMAN_TYPE_ABGR;

        } else {


            type = PIXMAN_TYPE_BGRA;


        }

    }

    return type;

}