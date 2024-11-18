static void spatial_compose53i(IDWTELEM *buffer, int width, int height, int stride){

    dwt_compose_t cs;

    spatial_compose53i_init(&cs, buffer, height, stride);

    while(cs.y <= height)

        spatial_compose53i_dy(&cs, buffer, width, height, stride);

}
