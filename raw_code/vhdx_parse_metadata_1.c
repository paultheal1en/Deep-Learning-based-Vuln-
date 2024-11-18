static int vhdx_parse_metadata(BlockDriverState *bs, BDRVVHDXState *s)

{

    int ret = 0;

    uint8_t *buffer;

    int offset = 0;

    uint32_t i = 0;

    VHDXMetadataTableEntry md_entry;



    buffer = qemu_blockalign(bs, VHDX_METADATA_TABLE_MAX_SIZE);



    ret = bdrv_pread(bs->file, s->metadata_rt.file_offset, buffer,

                     VHDX_METADATA_TABLE_MAX_SIZE);

    if (ret < 0) {

        goto exit;

    }

    memcpy(&s->metadata_hdr, buffer, sizeof(s->metadata_hdr));

    offset += sizeof(s->metadata_hdr);



    vhdx_metadata_header_le_import(&s->metadata_hdr);



    if (memcmp(&s->metadata_hdr.signature, "metadata", 8)) {

        ret = -EINVAL;

        goto exit;

    }



    s->metadata_entries.present = 0;



    if ((s->metadata_hdr.entry_count * sizeof(md_entry)) >

        (VHDX_METADATA_TABLE_MAX_SIZE - offset)) {

        ret = -EINVAL;

        goto exit;

    }



    for (i = 0; i < s->metadata_hdr.entry_count; i++) {

        memcpy(&md_entry, buffer + offset, sizeof(md_entry));

        offset += sizeof(md_entry);



        vhdx_metadata_entry_le_import(&md_entry);



        if (guid_eq(md_entry.item_id, file_param_guid)) {

            if (s->metadata_entries.present & META_FILE_PARAMETER_PRESENT) {

                ret = -EINVAL;

                goto exit;

            }

            s->metadata_entries.file_parameters_entry = md_entry;

            s->metadata_entries.present |= META_FILE_PARAMETER_PRESENT;

            continue;

        }



        if (guid_eq(md_entry.item_id, virtual_size_guid)) {

            if (s->metadata_entries.present & META_VIRTUAL_DISK_SIZE_PRESENT) {

                ret = -EINVAL;

                goto exit;

            }

            s->metadata_entries.virtual_disk_size_entry = md_entry;

            s->metadata_entries.present |= META_VIRTUAL_DISK_SIZE_PRESENT;

            continue;

        }



        if (guid_eq(md_entry.item_id, page83_guid)) {

            if (s->metadata_entries.present & META_PAGE_83_PRESENT) {

                ret = -EINVAL;

                goto exit;

            }

            s->metadata_entries.page83_data_entry = md_entry;

            s->metadata_entries.present |= META_PAGE_83_PRESENT;

            continue;

        }



        if (guid_eq(md_entry.item_id, logical_sector_guid)) {

            if (s->metadata_entries.present &

                META_LOGICAL_SECTOR_SIZE_PRESENT) {

                ret = -EINVAL;

                goto exit;

            }

            s->metadata_entries.logical_sector_size_entry = md_entry;

            s->metadata_entries.present |= META_LOGICAL_SECTOR_SIZE_PRESENT;

            continue;

        }



        if (guid_eq(md_entry.item_id, phys_sector_guid)) {

            if (s->metadata_entries.present & META_PHYS_SECTOR_SIZE_PRESENT) {

                ret = -EINVAL;

                goto exit;

            }

            s->metadata_entries.phys_sector_size_entry = md_entry;

            s->metadata_entries.present |= META_PHYS_SECTOR_SIZE_PRESENT;

            continue;

        }



        if (guid_eq(md_entry.item_id, parent_locator_guid)) {

            if (s->metadata_entries.present & META_PARENT_LOCATOR_PRESENT) {

                ret = -EINVAL;

                goto exit;

            }

            s->metadata_entries.parent_locator_entry = md_entry;

            s->metadata_entries.present |= META_PARENT_LOCATOR_PRESENT;

            continue;

        }



        if (md_entry.data_bits & VHDX_META_FLAGS_IS_REQUIRED) {

            /* cannot read vhdx file - required region table entry that

             * we do not understand.  per spec, we must fail to open */

            ret = -ENOTSUP;

            goto exit;

        }

    }



    if (s->metadata_entries.present != META_ALL_PRESENT) {

        ret = -ENOTSUP;

        goto exit;

    }



    ret = bdrv_pread(bs->file,

                     s->metadata_entries.file_parameters_entry.offset

                                         + s->metadata_rt.file_offset,

                     &s->params,

                     sizeof(s->params));



    if (ret < 0) {

        goto exit;

    }



    le32_to_cpus(&s->params.block_size);

    le32_to_cpus(&s->params.data_bits);





    /* We now have the file parameters, so we can tell if this is a

     * differencing file (i.e.. has_parent), is dynamic or fixed

     * sized (leave_blocks_allocated), and the block size */



    /* The parent locator required iff the file parameters has_parent set */

    if (s->params.data_bits & VHDX_PARAMS_HAS_PARENT) {

        if (s->metadata_entries.present & META_PARENT_LOCATOR_PRESENT) {

            /* TODO: parse  parent locator fields */

            ret = -ENOTSUP; /* temp, until differencing files are supported */

            goto exit;

        } else {

            /* if has_parent is set, but there is not parent locator present,

             * then that is an invalid combination */

            ret = -EINVAL;

            goto exit;

        }

    }



    /* determine virtual disk size, logical sector size,

     * and phys sector size */



    ret = bdrv_pread(bs->file,

                     s->metadata_entries.virtual_disk_size_entry.offset

                                           + s->metadata_rt.file_offset,

                     &s->virtual_disk_size,

                     sizeof(uint64_t));

    if (ret < 0) {

        goto exit;

    }

    ret = bdrv_pread(bs->file,

                     s->metadata_entries.logical_sector_size_entry.offset

                                             + s->metadata_rt.file_offset,

                     &s->logical_sector_size,

                     sizeof(uint32_t));

    if (ret < 0) {

        goto exit;

    }

    ret = bdrv_pread(bs->file,

                     s->metadata_entries.phys_sector_size_entry.offset

                                          + s->metadata_rt.file_offset,

                     &s->physical_sector_size,

                     sizeof(uint32_t));

    if (ret < 0) {

        goto exit;

    }



    le64_to_cpus(&s->virtual_disk_size);

    le32_to_cpus(&s->logical_sector_size);

    le32_to_cpus(&s->physical_sector_size);



    if (s->logical_sector_size == 0 || s->params.block_size == 0) {

        ret = -EINVAL;

        goto exit;

    }



    /* both block_size and sector_size are guaranteed powers of 2 */

    s->sectors_per_block = s->params.block_size / s->logical_sector_size;

    s->chunk_ratio = (VHDX_MAX_SECTORS_PER_BLOCK) *

                     (uint64_t)s->logical_sector_size /

                     (uint64_t)s->params.block_size;



    /* These values are ones we will want to use for division / multiplication

     * later on, and they are all guaranteed (per the spec) to be powers of 2,

     * so we can take advantage of that for shift operations during

     * reads/writes */

    if (s->logical_sector_size & (s->logical_sector_size - 1)) {

        ret = -EINVAL;

        goto exit;

    }

    if (s->sectors_per_block & (s->sectors_per_block - 1)) {

        ret = -EINVAL;

        goto exit;

    }

    if (s->chunk_ratio & (s->chunk_ratio - 1)) {

        ret = -EINVAL;

        goto exit;

    }

    s->block_size = s->params.block_size;

    if (s->block_size & (s->block_size - 1)) {

        ret = -EINVAL;

        goto exit;

    }



    vhdx_set_shift_bits(s);



    ret = 0;



exit:

    qemu_vfree(buffer);

    return ret;

}
