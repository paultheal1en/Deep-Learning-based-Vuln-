static int dmg_read_resource_fork(BlockDriverState *bs, DmgHeaderState *ds,

                                  uint64_t info_begin, uint64_t info_length)

{

    BDRVDMGState *s = bs->opaque;

    int ret;

    uint32_t count, rsrc_data_offset;

    uint8_t *buffer = NULL;

    uint64_t info_end;

    uint64_t offset;



    /* read offset from begin of resource fork (info_begin) to resource data */

    ret = read_uint32(bs, info_begin, &rsrc_data_offset);

    if (ret < 0) {

        goto fail;

    } else if (rsrc_data_offset > info_length) {

        ret = -EINVAL;

        goto fail;

    }



    /* read length of resource data */

    ret = read_uint32(bs, info_begin + 8, &count);

    if (ret < 0) {

        goto fail;

    } else if (count == 0 || rsrc_data_offset + count > info_length) {

        ret = -EINVAL;

        goto fail;

    }



    /* begin of resource data (consisting of one or more resources) */

    offset = info_begin + rsrc_data_offset;



    /* end of resource data (there is possibly a following resource map

     * which will be ignored). */

    info_end = offset + count;



    /* read offsets (mish blocks) from one or more resources in resource data */

    while (offset < info_end) {

        /* size of following resource */

        ret = read_uint32(bs, offset, &count);

        if (ret < 0) {

            goto fail;

        } else if (count == 0) {

            ret = -EINVAL;

            goto fail;

        }

        offset += 4;



        buffer = g_realloc(buffer, count);

        ret = bdrv_pread(bs->file, offset, buffer, count);

        if (ret < 0) {

            goto fail;

        }



        ret = dmg_read_mish_block(s, ds, buffer, count);

        if (ret < 0) {

            goto fail;

        }

        /* advance offset by size of resource */

        offset += count;

    }

    ret = 0;



fail:

    g_free(buffer);

    return ret;

}
