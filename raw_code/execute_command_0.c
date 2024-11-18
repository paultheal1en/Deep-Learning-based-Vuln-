static int execute_command(BlockDriverState *bdrv,

                           SCSIGenericReq *r, int direction,

                           BlockCompletionFunc *complete)

{

    r->io_header.interface_id = 'S';

    r->io_header.dxfer_direction = direction;

    r->io_header.dxferp = r->buf;

    r->io_header.dxfer_len = r->buflen;

    r->io_header.cmdp = r->req.cmd.buf;

    r->io_header.cmd_len = r->req.cmd.len;

    r->io_header.mx_sb_len = sizeof(r->req.sense);

    r->io_header.sbp = r->req.sense;

    r->io_header.timeout = MAX_UINT;

    r->io_header.usr_ptr = r;

    r->io_header.flags |= SG_FLAG_DIRECT_IO;



    r->req.aiocb = bdrv_aio_ioctl(bdrv, SG_IO, &r->io_header, complete, r);

    if (r->req.aiocb == NULL) {

        return -EIO;

    }



    return 0;

}
