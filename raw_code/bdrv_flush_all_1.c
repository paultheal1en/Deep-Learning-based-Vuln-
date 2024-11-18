int bdrv_flush_all(void)

{

    BlockDriverState *bs;

    int result = 0;



    QTAILQ_FOREACH(bs, &bdrv_states, device_list) {

        int ret = bdrv_flush(bs);

        if (ret < 0 && !result) {

            result = ret;

        }

    }



    return result;

}
