static ssize_t local_llistxattr(FsContext *ctx, const char *path,

                                void *value, size_t size)

{

    ssize_t retval;

    ssize_t actual_len = 0;

    char *orig_value, *orig_value_start;

    char *temp_value, *temp_value_start;

    ssize_t xattr_len, parsed_len = 0, attr_len;



    if (ctx->fs_sm != SM_MAPPED) {

        return llistxattr(rpath(ctx, path), value, size);

    }



    /* Get the actual len */

    xattr_len = llistxattr(rpath(ctx, path), value, 0);



    /* Now fetch the xattr and find the actual size */

    orig_value = qemu_malloc(xattr_len);

    xattr_len = llistxattr(rpath(ctx, path), orig_value, xattr_len);



    /*

     * For mapped security model drop user.virtfs namespace

     * from the list

     */

    temp_value = qemu_mallocz(xattr_len);

    temp_value_start = temp_value;

    orig_value_start = orig_value;

    while (xattr_len > parsed_len) {

        attr_len = strlen(orig_value) + 1;

        if (strncmp(orig_value, "user.virtfs.", 12) != 0) {

            /* Copy this entry */

            strcat(temp_value, orig_value);

            temp_value  += attr_len;

            actual_len += attr_len;

        }

        parsed_len += attr_len;

        orig_value += attr_len;

    }

    if (!size) {

        retval = actual_len;

        goto out;

    } else if (size >= actual_len) {

        /* now copy the parsed attribute list back */

        memset(value, 0, size);

        memcpy(value, temp_value_start, actual_len);

        retval = actual_len;

        goto out;

    }

    errno = ERANGE;

    retval = -1;

out:

    qemu_free(orig_value_start);

    qemu_free(temp_value_start);

    return retval;

}
