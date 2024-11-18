static int mp3info(void *data, int *byteSize, int *samplesPerFrame, int *sampleRate, int *isMono )

{

    uint8_t *dataTmp = (uint8_t *)data;

    uint32_t header = ( (uint32_t)dataTmp[0] << 24 ) | ( (uint32_t)dataTmp[1] << 16 ) | ( (uint32_t)dataTmp[2] << 8 ) | (uint32_t)dataTmp[3];

    int layerID = 3 - ((header >> 17) & 0x03);

    int bitRateID = ((header >> 12) & 0x0f);

    int sampleRateID = ((header >> 10) & 0x03);

    int bitRate = 0;

    int bitsPerSlot = sBitsPerSlot[layerID];

    int isPadded = ((header >> 9) & 0x01);



    if ( (( header >> 21 ) & 0x7ff) != 0x7ff ) {

        return 0;

    }



    if ( !isPadded ) {

        printf("Fatal error: mp3 data is not padded!\n");

        exit(0);

    }



    *isMono = ((header >>  6) & 0x03) == 0x03;



    if ( (header >> 19 ) & 0x01 ) {

        *sampleRate = sSampleRates[0][sampleRateID];

        bitRate = sBitRates[0][layerID][bitRateID] * 1000;

        *samplesPerFrame = sSamplesPerFrame[0][layerID];



    } else {

        if ( (header >> 20) & 0x01 ) {

            *sampleRate = sSampleRates[1][sampleRateID];

            bitRate = sBitRates[1][layerID][bitRateID] * 1000;

            *samplesPerFrame = sSamplesPerFrame[1][layerID];

        } else {

            *sampleRate = sSampleRates[2][sampleRateID];

            bitRate = sBitRates[1][layerID][bitRateID] * 1000;

            *samplesPerFrame = sSamplesPerFrame[2][layerID];

        }

    }



    *byteSize = ( ( ( ( *samplesPerFrame * (bitRate / bitsPerSlot) ) / *sampleRate ) + isPadded ) * bitsPerSlot);



    return 1;

}
