static int cdrom_probe_device(const char *filename)

{

    if (strstart(filename, "/dev/cd", NULL))

        return 100;

    return 0;

}
