static inline uint32_t vmsvga_fifo_read_raw(struct vmsvga_state_s *s)

{

    uint32_t cmd = s->fifo[CMD(stop) >> 2];



    s->cmd->stop = cpu_to_le32(CMD(stop) + 4);

    if (CMD(stop) >= CMD(max)) {

        s->cmd->stop = s->cmd->min;

    }

    return cmd;

}
