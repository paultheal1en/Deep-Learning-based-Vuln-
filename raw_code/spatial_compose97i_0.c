static void spatial_compose97i(IDWTELEM *buffer, int width, int height, int stride){

    dwt_compose_t cs;

    spatial_compose97i_init(&cs, buffer, height, stride);

    while(cs.y <= height)

        spatial_compose97i_dy(&cs, buffer, width, height, stride);

}
