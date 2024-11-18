static int vmdk_open_vmdk4(BlockDriverState *bs,

                           BlockDriverState *file,

                           int flags, Error **errp)

{

    int ret;

    uint32_t magic;

    uint32_t l1_size, l1_entry_sectors;

    VMDK4Header header;

    VmdkExtent *extent;

    BDRVVmdkState *s = bs->opaque;

    int64_t l1_backup_offset = 0;



    ret = bdrv_pread(file, sizeof(magic), &header, sizeof(header));

    if (ret < 0) {

        error_setg_errno(errp, -ret,

                         "Could not read header from file '%s'",

                         file->filename);

        return -EINVAL;

    }

    if (header.capacity == 0) {

        uint64_t desc_offset = le64_to_cpu(header.desc_offset);

        if (desc_offset) {

            char *buf = vmdk_read_desc(file, desc_offset << 9, errp);

            if (!buf) {

                return -EINVAL;

            }

            ret = vmdk_open_desc_file(bs, flags, buf, errp);

            g_free(buf);

            return ret;

        }

    }



    if (!s->create_type) {

        s->create_type = g_strdup("monolithicSparse");

    }



    if (le64_to_cpu(header.gd_offset) == VMDK4_GD_AT_END) {

        /*

         * The footer takes precedence over the header, so read it in. The

         * footer starts at offset -1024 from the end: One sector for the

         * footer, and another one for the end-of-stream marker.

         */

        struct {

            struct {

                uint64_t val;

                uint32_t size;

                uint32_t type;

                uint8_t pad[512 - 16];

            } QEMU_PACKED footer_marker;



            uint32_t magic;

            VMDK4Header header;

            uint8_t pad[512 - 4 - sizeof(VMDK4Header)];



            struct {

                uint64_t val;

                uint32_t size;

                uint32_t type;

                uint8_t pad[512 - 16];

            } QEMU_PACKED eos_marker;

        } QEMU_PACKED footer;



        ret = bdrv_pread(file,

            bs->file->total_sectors * 512 - 1536,

            &footer, sizeof(footer));

        if (ret < 0) {

            error_setg_errno(errp, -ret, "Failed to read footer");

            return ret;

        }



        /* Some sanity checks for the footer */

        if (be32_to_cpu(footer.magic) != VMDK4_MAGIC ||

            le32_to_cpu(footer.footer_marker.size) != 0  ||

            le32_to_cpu(footer.footer_marker.type) != MARKER_FOOTER ||

            le64_to_cpu(footer.eos_marker.val) != 0  ||

            le32_to_cpu(footer.eos_marker.size) != 0  ||

            le32_to_cpu(footer.eos_marker.type) != MARKER_END_OF_STREAM)

        {

            error_setg(errp, "Invalid footer");

            return -EINVAL;

        }



        header = footer.header;

    }



    if (le32_to_cpu(header.version) > 3) {

        char buf[64];

        snprintf(buf, sizeof(buf), "VMDK version %" PRId32,

                 le32_to_cpu(header.version));

        error_set(errp, QERR_UNKNOWN_BLOCK_FORMAT_FEATURE,

                  bdrv_get_device_or_node_name(bs), "vmdk", buf);

        return -ENOTSUP;

    } else if (le32_to_cpu(header.version) == 3 && (flags & BDRV_O_RDWR)) {

        /* VMware KB 2064959 explains that version 3 added support for

         * persistent changed block tracking (CBT), and backup software can

         * read it as version=1 if it doesn't care about the changed area

         * information. So we are safe to enable read only. */

        error_setg(errp, "VMDK version 3 must be read only");

        return -EINVAL;

    }



    if (le32_to_cpu(header.num_gtes_per_gt) > 512) {

        error_setg(errp, "L2 table size too big");

        return -EINVAL;

    }



    l1_entry_sectors = le32_to_cpu(header.num_gtes_per_gt)

                        * le64_to_cpu(header.granularity);

    if (l1_entry_sectors == 0) {

        error_setg(errp, "L1 entry size is invalid");

        return -EINVAL;

    }

    l1_size = (le64_to_cpu(header.capacity) + l1_entry_sectors - 1)

                / l1_entry_sectors;

    if (le32_to_cpu(header.flags) & VMDK4_FLAG_RGD) {

        l1_backup_offset = le64_to_cpu(header.rgd_offset) << 9;

    }

    if (bdrv_nb_sectors(file) < le64_to_cpu(header.grain_offset)) {

        error_setg(errp, "File truncated, expecting at least %" PRId64 " bytes",

                   (int64_t)(le64_to_cpu(header.grain_offset)

                             * BDRV_SECTOR_SIZE));

        return -EINVAL;

    }



    ret = vmdk_add_extent(bs, file, false,

                          le64_to_cpu(header.capacity),

                          le64_to_cpu(header.gd_offset) << 9,

                          l1_backup_offset,

                          l1_size,

                          le32_to_cpu(header.num_gtes_per_gt),

                          le64_to_cpu(header.granularity),

                          &extent,

                          errp);

    if (ret < 0) {

        return ret;

    }

    extent->compressed =

        le16_to_cpu(header.compressAlgorithm) == VMDK4_COMPRESSION_DEFLATE;

    if (extent->compressed) {

        g_free(s->create_type);

        s->create_type = g_strdup("streamOptimized");

    }

    extent->has_marker = le32_to_cpu(header.flags) & VMDK4_FLAG_MARKER;

    extent->version = le32_to_cpu(header.version);

    extent->has_zero_grain = le32_to_cpu(header.flags) & VMDK4_FLAG_ZERO_GRAIN;

    ret = vmdk_init_tables(bs, extent, errp);

    if (ret) {

        /* free extent allocated by vmdk_add_extent */

        vmdk_free_last_extent(bs);

    }

    return ret;

}
