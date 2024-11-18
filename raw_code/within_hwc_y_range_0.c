static int within_hwc_y_range(SM501State *state, int y, int crt)

{

    int hwc_y = get_hwc_y(state, crt);

    return (hwc_y <= y && y < hwc_y + SM501_HWC_HEIGHT);

}
