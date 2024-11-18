char *qemu_strdup(const char *str)

{

    char *ptr;

    size_t len = strlen(str);

    ptr = qemu_malloc(len + 1);

    if (!ptr)

        return NULL;

    pstrcpy(ptr, len + 1, str);

    return ptr;

}
