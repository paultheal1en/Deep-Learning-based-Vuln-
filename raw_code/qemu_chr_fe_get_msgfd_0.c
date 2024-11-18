int qemu_chr_fe_get_msgfd(CharDriverState *s)

{

    int fd;

    return (qemu_chr_fe_get_msgfds(s, &fd, 1) == 1) ? fd : -1;

}
