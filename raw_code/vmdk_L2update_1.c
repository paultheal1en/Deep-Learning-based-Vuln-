static int vmdk_L2update(BlockDriverState *bs, VmdkMetaData *m_data)

{

    BDRVVmdkState *s = bs->opaque;



    /* update L2 table */

    if (bdrv_pwrite(bs->file, ((int64_t)m_data->l2_offset * 512) + (m_data->l2_index * sizeof(m_data->offset)),

                    &(m_data->offset), sizeof(m_data->offset)) != sizeof(m_data->offset))

        return -1;

    /* update backup L2 table */

    if (s->l1_backup_table_offset != 0) {

        m_data->l2_offset = s->l1_backup_table[m_data->l1_index];

        if (bdrv_pwrite(bs->file, ((int64_t)m_data->l2_offset * 512) + (m_data->l2_index * sizeof(m_data->offset)),

                        &(m_data->offset), sizeof(m_data->offset)) != sizeof(m_data->offset))

            return -1;

    }



    return 0;

}
