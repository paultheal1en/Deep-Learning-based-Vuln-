static void sd_response_r1_make(SDState *sd, uint8_t *response)

{

    uint32_t status = sd->card_status;

    /* Clear the "clear on read" status bits (except APP_CMD) */

    sd->card_status &= ~CARD_STATUS_C | APP_CMD;



    response[0] = (status >> 24) & 0xff;

    response[1] = (status >> 16) & 0xff;

    response[2] = (status >> 8) & 0xff;

    response[3] = (status >> 0) & 0xff;

}
