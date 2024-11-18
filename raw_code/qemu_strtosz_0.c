int64_t qemu_strtosz(const char *nptr, char **end)

{

    return do_strtosz(nptr, end, 'B', 1024);

}
