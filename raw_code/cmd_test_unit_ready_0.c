static void cmd_test_unit_ready(IDEState *s, uint8_t *buf)

{

    if (bdrv_is_inserted(s->bs)) {

        ide_atapi_cmd_ok(s);

    } else {

        ide_atapi_cmd_error(s, SENSE_NOT_READY, ASC_MEDIUM_NOT_PRESENT);

    }

}
