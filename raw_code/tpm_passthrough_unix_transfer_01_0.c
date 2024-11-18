static int tpm_passthrough_unix_transfer(int tpm_fd,

                                         const TPMLocality *locty_data)

{

    return tpm_passthrough_unix_tx_bufs(tpm_fd,

                                        locty_data->w_buffer.buffer,

                                        locty_data->w_offset,

                                        locty_data->r_buffer.buffer,

                                        locty_data->r_buffer.size);

}
