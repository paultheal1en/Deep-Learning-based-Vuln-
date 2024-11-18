static int vmdk_open_desc_file(BlockDriverState *bs, int flags, char *buf,

                               Error **errp)

{

    int ret;

    char ct[128];

    BDRVVmdkState *s = bs->opaque;



    if (vmdk_parse_description(buf, "createType", ct, sizeof(ct))) {

        error_setg(errp, "invalid VMDK image descriptor");

        ret = -EINVAL;

        goto exit;

    }

    if (strcmp(ct, "monolithicFlat") &&

        strcmp(ct, "vmfs") &&

        strcmp(ct, "vmfsSparse") &&

        strcmp(ct, "twoGbMaxExtentSparse") &&

        strcmp(ct, "twoGbMaxExtentFlat")) {

        error_setg(errp, "Unsupported image type '%s'", ct);

        ret = -ENOTSUP;

        goto exit;

    }

    s->create_type = g_strdup(ct);

    s->desc_offset = 0;

    ret = vmdk_parse_extents(buf, bs, bs->file->exact_filename, errp);

exit:

    return ret;

}
