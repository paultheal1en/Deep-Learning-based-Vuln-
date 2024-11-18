static uint32_t read_u32(uint8_t *data, size_t offset)

{

    return ((data[offset] << 24) | (data[offset + 1] << 16) |

	    (data[offset + 2] << 8) | data[offset + 3]);

}
