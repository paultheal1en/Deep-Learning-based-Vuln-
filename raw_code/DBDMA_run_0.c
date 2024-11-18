static void DBDMA_run (DBDMA_channel *ch)

{

    int channel;



    for (channel = 0; channel < DBDMA_CHANNELS; channel++, ch++) {

            uint32_t status = be32_to_cpu(ch->regs[DBDMA_STATUS]);

            if (!ch->processing && (status & RUN) && (status & ACTIVE))

                channel_run(ch);

    }

}
