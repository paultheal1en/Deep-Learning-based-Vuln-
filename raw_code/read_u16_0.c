static uint16_t read_u16(uint8_t *data, size_t offset)

{

    return ((data[offset] & 0xFF) << 8) | (data[offset + 1] & 0xFF);

}
