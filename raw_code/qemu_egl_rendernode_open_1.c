int qemu_egl_rendernode_open(void)

{

    DIR *dir;

    struct dirent *e;

    int r, fd;

    char *p;



    dir = opendir("/dev/dri");

    if (!dir) {

        return -1;

    }



    fd = -1;

    while ((e = readdir(dir))) {

        if (e->d_type != DT_CHR) {

            continue;

        }



        if (strncmp(e->d_name, "renderD", 7)) {

            continue;

        }



        r = asprintf(&p, "/dev/dri/%s", e->d_name);

        if (r < 0) {

            return -1;

        }



        r = open(p, O_RDWR | O_CLOEXEC | O_NOCTTY | O_NONBLOCK);

        if (r < 0) {

            free(p);

            continue;

        }

        fd = r;

        free(p);

        break;

    }



    closedir(dir);

    if (fd < 0) {

        return -1;

    }

    return fd;

}
