static int ftp_conn_control_block_control(void *data)

{

    FTPContext *s = data;

    return s->conn_control_block_flag;

}
