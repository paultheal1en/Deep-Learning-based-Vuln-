void show_banner(void)

{

    fprintf(stderr, "%s version " FFMPEG_VERSION ", Copyright (c) %d-%d the FFmpeg developers\n",

            program_name, program_birth_year, this_year);

    fprintf(stderr, "  built on %s %s with %s %s\n",

            __DATE__, __TIME__, CC_TYPE, CC_VERSION);

    fprintf(stderr, "  configuration: " FFMPEG_CONFIGURATION "\n");

    print_all_libs_info(stderr, INDENT|SHOW_CONFIG);

    print_all_libs_info(stderr, INDENT|SHOW_VERSION);

}
