static void opt_pad_color(const char *arg) {

    /* Input is expected to be six hex digits similar to

       how colors are expressed in html tags (but without the #) */

    int rgb = strtol(arg, NULL, 16);

    int r,g,b;



    r = (rgb >> 16);

    g = ((rgb >> 8) & 255);

    b = (rgb & 255);



    padcolor[0] = RGB_TO_Y(r,g,b);

    padcolor[1] = RGB_TO_U(r,g,b,0);

    padcolor[2] = RGB_TO_V(r,g,b,0);

}
