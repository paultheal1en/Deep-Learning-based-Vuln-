static inline int vmsvga_fifo_length(struct vmsvga_state_s *s)

{

    int num;



    if (!s->config || !s->enable) {

        return 0;

    }



    /* Check range and alignment.  */

    if ((CMD(min) | CMD(max) | CMD(next_cmd) | CMD(stop)) & 3) {

        return 0;

    }

    if (CMD(min) < (uint8_t *) s->cmd->fifo - (uint8_t *) s->fifo) {

        return 0;

    }

    if (CMD(max) > SVGA_FIFO_SIZE ||

        CMD(min) >= SVGA_FIFO_SIZE ||

        CMD(stop) >= SVGA_FIFO_SIZE ||

        CMD(next_cmd) >= SVGA_FIFO_SIZE) {

        return 0;

    }

    if (CMD(max) < CMD(min) + 10 * 1024) {

        return 0;

    }



    num = CMD(next_cmd) - CMD(stop);

    if (num < 0) {

        num += CMD(max) - CMD(min);

    }

    return num >> 2;

}