static int qemu_rdma_reg_whole_ram_blocks(RDMAContext *rdma)

{

    int i;

    RDMALocalBlocks *local = &rdma->local_ram_blocks;



    for (i = 0; i < local->nb_blocks; i++) {

        local->block[i].mr =

            ibv_reg_mr(rdma->pd,

                    local->block[i].local_host_addr,

                    local->block[i].length,

                    IBV_ACCESS_LOCAL_WRITE |

                    IBV_ACCESS_REMOTE_WRITE

                    );

        if (!local->block[i].mr) {

            perror("Failed to register local dest ram block!\n");

            break;

        }

        rdma->total_registrations++;

    }



    if (i >= local->nb_blocks) {

        return 0;

    }



    for (i--; i >= 0; i--) {

        ibv_dereg_mr(local->block[i].mr);

        rdma->total_registrations--;

    }



    return -1;



}
