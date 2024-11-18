static void vhdx_parse_header(BlockDriverState *bs, BDRVVHDXState *s,

                              Error **errp)

{

    int ret;

    VHDXHeader *header1;

    VHDXHeader *header2;

    bool h1_valid = false;

    bool h2_valid = false;

    uint64_t h1_seq = 0;

    uint64_t h2_seq = 0;

    uint8_t *buffer;



    /* header1 & header2 are freed in vhdx_close() */

    header1 = qemu_blockalign(bs, sizeof(VHDXHeader));

    header2 = qemu_blockalign(bs, sizeof(VHDXHeader));



    buffer = qemu_blockalign(bs, VHDX_HEADER_SIZE);



    s->headers[0] = header1;

    s->headers[1] = header2;



    /* We have to read the whole VHDX_HEADER_SIZE instead of

     * sizeof(VHDXHeader), because the checksum is over the whole

     * region */

    ret = bdrv_pread(bs->file, VHDX_HEADER1_OFFSET, buffer, VHDX_HEADER_SIZE);

    if (ret < 0) {

        goto fail;

    }

    /* copy over just the relevant portion that we need */

    memcpy(header1, buffer, sizeof(VHDXHeader));

    vhdx_header_le_import(header1);



    if (vhdx_checksum_is_valid(buffer, VHDX_HEADER_SIZE, 4) &&

        !memcmp(&header1->signature, "head", 4)             &&

        header1->version == 1) {

        h1_seq = header1->sequence_number;

        h1_valid = true;

    }



    ret = bdrv_pread(bs->file, VHDX_HEADER2_OFFSET, buffer, VHDX_HEADER_SIZE);

    if (ret < 0) {

        goto fail;

    }

    /* copy over just the relevant portion that we need */

    memcpy(header2, buffer, sizeof(VHDXHeader));

    vhdx_header_le_import(header2);



    if (vhdx_checksum_is_valid(buffer, VHDX_HEADER_SIZE, 4) &&

        !memcmp(&header2->signature, "head", 4)             &&

        header2->version == 1) {

        h2_seq = header2->sequence_number;

        h2_valid = true;

    }



    /* If there is only 1 valid header (or no valid headers), we

     * don't care what the sequence numbers are */

    if (h1_valid && !h2_valid) {

        s->curr_header = 0;

    } else if (!h1_valid && h2_valid) {

        s->curr_header = 1;

    } else if (!h1_valid && !h2_valid) {

        goto fail;

    } else {

        /* If both headers are valid, then we choose the active one by the

         * highest sequence number.  If the sequence numbers are equal, that is

         * invalid */

        if (h1_seq > h2_seq) {

            s->curr_header = 0;

        } else if (h2_seq > h1_seq) {

            s->curr_header = 1;

        } else {

            goto fail;

        }

    }



    vhdx_region_register(s, s->headers[s->curr_header]->log_offset,

                            s->headers[s->curr_header]->log_length);

    goto exit;



fail:

    error_setg_errno(errp, -ret, "No valid VHDX header found");

    qemu_vfree(header1);

    qemu_vfree(header2);

    s->headers[0] = NULL;

    s->headers[1] = NULL;

exit:

    qemu_vfree(buffer);

}
