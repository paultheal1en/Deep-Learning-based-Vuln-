static int execute_command(BlockDriverState *bdrv,

                           SCSIRequest *r, int direction,

			   BlockDriverCompletionFunc *complete)

{



    r->io_header.interface_id = 'S';

    r->io_header.dxfer_direction = direction;

    r->io_header.dxferp = r->buf;

    r->io_header.dxfer_len = r->buflen;

    r->io_header.cmdp = r->cmd;

    r->io_header.cmd_len = r->cmdlen;

    r->io_header.mx_sb_len = sizeof(r->dev->sensebuf);

    r->io_header.sbp = r->dev->sensebuf;

    r->io_header.timeout = MAX_UINT;

    r->io_header.usr_ptr = r;

    r->io_header.flags |= SG_FLAG_DIRECT_IO;



    if (bdrv_pwrite(bdrv, -1, &r->io_header, sizeof(r->io_header)) == -1) {

        BADF("execute_command: write failed ! (%d)\n", errno);

        return -1;

    }

    if (complete == NULL) {

        int ret;

        r->aiocb = NULL;

        while ((ret = bdrv_pread(bdrv, -1, &r->io_header,

                                           sizeof(r->io_header))) == -1 &&

                      errno == EINTR);

        if (ret == -1) {

            BADF("execute_command: read failed !\n");

            return -1;

        }

        return 0;

    }



    r->aiocb = bdrv_aio_read(bdrv, 0, (uint8_t*)&r->io_header,

                          -(int64_t)sizeof(r->io_header), complete, r);

    if (r->aiocb == NULL) {

        BADF("execute_command: read failed !\n");

        return -1;

    }



    return 0;

}
