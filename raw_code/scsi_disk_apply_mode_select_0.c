static void scsi_disk_apply_mode_select(SCSIDiskState *s, int page, uint8_t *p)

{

    switch (page) {

    case MODE_PAGE_CACHING:

        bdrv_set_enable_write_cache(s->qdev.conf.bs, (p[0] & 4) != 0);

        break;



    default:

        break;

    }

}
