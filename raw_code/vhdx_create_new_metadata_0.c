static int vhdx_create_new_metadata(BlockDriverState *bs,

                                    uint64_t image_size,

                                    uint32_t block_size,

                                    uint32_t sector_size,

                                    uint64_t metadata_offset,

                                    VHDXImageType type)

{

    int ret = 0;

    uint32_t offset = 0;

    void *buffer = NULL;

    void *entry_buffer;

    VHDXMetadataTableHeader *md_table;

    VHDXMetadataTableEntry  *md_table_entry;



    /* Metadata entries */

    VHDXFileParameters     *mt_file_params;

    VHDXVirtualDiskSize    *mt_virtual_size;

    VHDXPage83Data         *mt_page83;

    VHDXVirtualDiskLogicalSectorSize  *mt_log_sector_size;

    VHDXVirtualDiskPhysicalSectorSize *mt_phys_sector_size;



    entry_buffer = g_malloc0(VHDX_METADATA_ENTRY_BUFFER_SIZE);



    mt_file_params = entry_buffer;

    offset += sizeof(VHDXFileParameters);

    mt_virtual_size = entry_buffer + offset;

    offset += sizeof(VHDXVirtualDiskSize);

    mt_page83 = entry_buffer + offset;

    offset += sizeof(VHDXPage83Data);

    mt_log_sector_size = entry_buffer + offset;

    offset += sizeof(VHDXVirtualDiskLogicalSectorSize);

    mt_phys_sector_size = entry_buffer + offset;



    mt_file_params->block_size = cpu_to_le32(block_size);

    if (type == VHDX_TYPE_FIXED) {

        mt_file_params->data_bits |= VHDX_PARAMS_LEAVE_BLOCKS_ALLOCED;

        cpu_to_le32s(&mt_file_params->data_bits);

    }



    vhdx_guid_generate(&mt_page83->page_83_data);

    cpu_to_leguids(&mt_page83->page_83_data);

    mt_virtual_size->virtual_disk_size        = cpu_to_le64(image_size);

    mt_log_sector_size->logical_sector_size   = cpu_to_le32(sector_size);

    mt_phys_sector_size->physical_sector_size = cpu_to_le32(sector_size);



    buffer = g_malloc0(VHDX_HEADER_BLOCK_SIZE);

    md_table = buffer;



    md_table->signature   = VHDX_METADATA_SIGNATURE;

    md_table->entry_count = 5;

    vhdx_metadata_header_le_export(md_table);





    /* This will reference beyond the reserved table portion */

    offset = 64 * KiB;



    md_table_entry = buffer + sizeof(VHDXMetadataTableHeader);



    md_table_entry[0].item_id = file_param_guid;

    md_table_entry[0].offset  = offset;

    md_table_entry[0].length  = sizeof(VHDXFileParameters);

    md_table_entry[0].data_bits |= VHDX_META_FLAGS_IS_REQUIRED;

    offset += md_table_entry[0].length;

    vhdx_metadata_entry_le_export(&md_table_entry[0]);



    md_table_entry[1].item_id = virtual_size_guid;

    md_table_entry[1].offset  = offset;

    md_table_entry[1].length  = sizeof(VHDXVirtualDiskSize);

    md_table_entry[1].data_bits |= VHDX_META_FLAGS_IS_REQUIRED |

                                   VHDX_META_FLAGS_IS_VIRTUAL_DISK;

    offset += md_table_entry[1].length;

    vhdx_metadata_entry_le_export(&md_table_entry[1]);



    md_table_entry[2].item_id = page83_guid;

    md_table_entry[2].offset  = offset;

    md_table_entry[2].length  = sizeof(VHDXPage83Data);

    md_table_entry[2].data_bits |= VHDX_META_FLAGS_IS_REQUIRED |

                                   VHDX_META_FLAGS_IS_VIRTUAL_DISK;

    offset += md_table_entry[2].length;

    vhdx_metadata_entry_le_export(&md_table_entry[2]);



    md_table_entry[3].item_id = logical_sector_guid;

    md_table_entry[3].offset  = offset;

    md_table_entry[3].length  = sizeof(VHDXVirtualDiskLogicalSectorSize);

    md_table_entry[3].data_bits |= VHDX_META_FLAGS_IS_REQUIRED |

                                   VHDX_META_FLAGS_IS_VIRTUAL_DISK;

    offset += md_table_entry[3].length;

    vhdx_metadata_entry_le_export(&md_table_entry[3]);



    md_table_entry[4].item_id = phys_sector_guid;

    md_table_entry[4].offset  = offset;

    md_table_entry[4].length  = sizeof(VHDXVirtualDiskPhysicalSectorSize);

    md_table_entry[4].data_bits |= VHDX_META_FLAGS_IS_REQUIRED |

                                   VHDX_META_FLAGS_IS_VIRTUAL_DISK;

    vhdx_metadata_entry_le_export(&md_table_entry[4]);



    ret = bdrv_pwrite(bs, metadata_offset, buffer, VHDX_HEADER_BLOCK_SIZE);

    if (ret < 0) {

        goto exit;

    }



    ret = bdrv_pwrite(bs, metadata_offset + (64 * KiB), entry_buffer,

                      VHDX_METADATA_ENTRY_BUFFER_SIZE);

    if (ret < 0) {

        goto exit;

    }





exit:

    g_free(buffer);

    g_free(entry_buffer);

    return ret;

}
