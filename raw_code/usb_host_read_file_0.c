static int usb_host_read_file(char *line, size_t line_size, const char *device_file, const char *device_name)

{

    FILE *f;

    int ret = 0;

    char filename[PATH_MAX];



    snprintf(filename, PATH_MAX, device_file, device_name);

    f = fopen(filename, "r");

    if (f) {

        fgets(line, line_size, f);

        fclose(f);

        ret = 1;

    } else {

        term_printf("husb: could not open %s\n", filename);

    }



    return ret;

}
