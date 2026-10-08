//
// RT64
//

#include "rt64_rdp_tmem.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <cinttypes>

#include "xxHash/xxh3.h"

#include "common/rt64_tmem_hasher.h"

#include "rt64_state.h"

namespace RT64 {
    // Rice/GLideN64 texture key ("crc#fmt#siz[#palettecrc]", lowercase hex), computed from RDRAM the same way
    // tools/texture_hasher derives it from a texture dump. Lets hi-res packs named in the Rice format be used
    // without an rt64.json mapping. The hashing below follows GLideN64 and RiceVideo (GPL), as in texture_hasher.
    static std::string computeRiceKey(const State *state, const LoadTile &drawTile, uint16_t drawWidth, uint16_t drawHeight, uint32_t tlut) {
        const uint32_t RDRAMBytes = RDRAMSize + 1;
        const LoadOperation &loadOp = state->rdp->rice.lastLoadOpByTMEM[drawTile.tmem];
        const uint8_t *RDRAM = reinterpret_cast<const uint8_t *>(state->RDRAM);

        auto calculateDXT = [](uint32_t txl2words) {
            return (txl2words == 0) ? 1U : (2048 + txl2words - 1) / txl2words;
        };

        auto txl2Words = [](uint32_t width, uint32_t size) {
            static const uint32_t sizeBytes[4] = { 0, 1, 2, 4 };
            return (size == 0) ? std::max(1U, width / 16) : std::max(1U, width * sizeBytes[size] / 8);
        };

        auto reverseDXT = [&](uint32_t val, uint32_t width, uint32_t size) {
            if (val == 0x800) {
                return 1;
            }

            int low = 2047 / val;
            if (calculateDXT(low) > val) {
                low++;
            }

            int high = 2047 / (val - 1);
            if (low == high) {
                return low;
            }

            for (int i = low; i <= high; i++) {
                if (txl2Words(width, size) == uint32_t(i)) {
                    return i;
                }
            }

            return (low + high) / 2;
        };

        auto riceCRC32 = [](const uint8_t *src, int width, int height, int size, int rowStride) {
            uint32_t crc = 0;
            const int bytesPerLine = width << size >> 1;
            for (int y = height - 1; y >= 0; y--) {
                uint32_t esi = 0;
                for (int x = bytesPerLine - 4; x >= 0; x -= 4) {
                    memcpy(&esi, src + x, sizeof(esi));
                    esi ^= x;
                    crc = (crc << 4) + ((crc >> 28) & 15);
                    crc += esi;
                }

                esi ^= y;
                crc += esi;
                src += rowStride;
            }

            return crc;
        };

        // Width, height and bytes per line of the texture as Rice sees them.
        int bpl, width, height;
        uint32_t rdramStart;
        const uint32_t commonBytesOffset = (loadOp.tile.uls >> 2) << loadOp.texture.siz >> 1;
        const uint32_t commonBytesPerRow = loadOp.texture.width << loadOp.texture.siz >> 1;
        if (loadOp.type == LoadOperation::Type::Tile) {
            uint16_t tileWidth = (std::max((loadOp.tile.lrs >> 2) - (loadOp.tile.uls >> 2), 0) + 1) & 0x03FF;
            uint16_t tileHeight = (std::max((loadOp.tile.lrt >> 2) - (loadOp.tile.ult >> 2), 0) + 1) & 0x03FF;
            tileWidth = (loadOp.tile.masks != 0) ? std::min(tileWidth, uint16_t(1U << loadOp.tile.masks)) : tileWidth;
            bpl = loadOp.texture.width << loadOp.texture.siz >> 1;
            width = std::min(tileWidth, loadOp.texture.width);
            if (loadOp.tile.siz > drawTile.siz) {
                width <<= loadOp.tile.siz - drawTile.siz;
            }

            height = (loadOp.tile.maskt != 0) ? std::min(tileHeight, uint16_t(1U << loadOp.tile.maskt)) : tileHeight;
            rdramStart = loadOp.texture.address + commonBytesOffset + commonBytesPerRow * (loadOp.tile.ult >> 2);
        }
        else if (loadOp.type == LoadOperation::Type::Block) {
            const int tileWidth = std::max((drawTile.lrs >> 2) - (drawTile.uls >> 2), 0) + 1;
            const int tileHeight = std::max((drawTile.lrt >> 2) - (drawTile.ult >> 2), 0) + 1;
            const int maskWidth = (drawTile.masks == 0) ? tileWidth : (1 << drawTile.masks);
            const int maskHeight = (drawTile.maskt == 0) ? tileHeight : (1 << drawTile.maskt);
            const bool clampS = (drawTile.masks == 0) || (drawTile.cms & G_TX_CLAMP);
            const bool clampT = (drawTile.maskt == 0) || (drawTile.cmt & G_TX_CLAMP);
            width = (clampS && (tileWidth <= 256)) ? std::min(maskWidth, tileWidth) : maskWidth;
            height = ((clampT && (tileHeight <= 256)) || (maskHeight > 256)) ? std::min(maskHeight, tileHeight) : maskHeight;
            if (drawTile.siz == G_IM_SIZ_32b) {
                bpl = drawTile.line << 4;
            }
            else if (loadOp.tile.lrt == 0) {
                bpl = drawTile.line << 3;
            }
            else {
                uint32_t dxt = loadOp.tile.lrt;
                if (dxt > 1) {
                    dxt = reverseDXT(dxt, loadOp.texture.width, loadOp.texture.siz);
                }

                bpl = dxt << 3;
            }

            rdramStart = loadOp.texture.address + commonBytesOffset + commonBytesPerRow * loadOp.tile.ult;
        }
        else {
            return std::string();
        }

        if ((width <= 0) || (height <= 0) || (bpl <= 0)) {
            return std::string();
        }

        // Rice hashes at least this many bytes, even if it leaks into other parts of RAM.
        const uint64_t expectedSize = uint64_t(height - 1) * bpl + (uint32_t(width) << drawTile.siz >> 1);
        if ((uint64_t(rdramStart) + expectedSize) > RDRAMBytes) {
            return std::string();
        }

        const uint8_t *src = &RDRAM[rdramStart];
        char key[48];
        int keyLength = snprintf(key, sizeof(key), "%08x#%u#%u", riceCRC32(src, width, height, drawTile.siz, bpl), uint32_t(drawTile.fmt), uint32_t(drawTile.siz));
        if (tlut > 0) {
            // Palette CRC covers the entries up to the highest index the texture uses.
            uint32_t ciMax = 0;
            if (drawTile.siz == G_IM_SIZ_4b) {
                for (int y = 0; (y < height) && (ciMax < 0xF); y++) {
                    const uint8_t *row = src + bpl * y;
                    for (int x = 0; x < (width >> 1); x++) {
                        ciMax = std::max({ ciMax, uint32_t(row[x] >> 4), uint32_t(row[x] & 0xF) });
                    }
                }
            }
            else {
                for (int y = 0; (y < height) && (ciMax < 0xFF); y++) {
                    const uint8_t *row = src + bpl * y;
                    for (int x = 0; x < width; x++) {
                        ciMax = std::max(ciMax, uint32_t(row[x]));
                    }
                }
            }

            const bool CI4 = (drawTile.siz == G_IM_SIZ_4b);
            const int32_t paletteTMEM = (RDP_TMEM_WORDS >> 1) + (CI4 ? (drawTile.palette << 4) : 0);
            const LoadOperation &paletteLoadOp = state->rdp->rice.lastLoadOpByTMEM[paletteTMEM];
            const uint32_t paletteBytesOffset = (paletteLoadOp.tile.uls >> 2) << paletteLoadOp.texture.siz >> 1;
            const uint32_t paletteBytesPerRow = paletteLoadOp.texture.width << paletteLoadOp.texture.siz >> 1;
            const uint32_t paletteStart = paletteLoadOp.texture.address + paletteBytesOffset + paletteBytesPerRow * (paletteLoadOp.tile.ult >> 2);
            const uint32_t paletteBytes = (ciMax + 1) * 2;
            if ((uint64_t(paletteStart) + paletteBytes) > RDRAMBytes) {
                return std::string();
            }

            const uint32_t paletteCrc = riceCRC32(&RDRAM[paletteStart], ciMax + 1, 1, 2, CI4 ? 32 : 512);
            keyLength += snprintf(key + keyLength, sizeof(key) - keyLength, "#%08x", paletteCrc);
        }

        return std::string(key, keyLength);
    }

    static void md5(const uint8_t *data, size_t size, uint8_t digest[16]) {
        static const uint32_t K[64] = {
            0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
            0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
            0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
            0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
            0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
            0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
            0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
            0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391 };
        static const uint32_t S[64] = {
            7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
            4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21 };

        std::vector<uint8_t> msg(data, data + size);
        msg.push_back(0x80);
        while ((msg.size() % 64) != 56) {
            msg.push_back(0);
        }

        const uint64_t bits = uint64_t(size) * 8;
        for (int i = 0; i < 8; i++) {
            msg.push_back(uint8_t(bits >> (8 * i)));
        }

        uint32_t h[4] = { 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476 };
        for (size_t chunk = 0; chunk < msg.size(); chunk += 64) {
            uint32_t w[16];
            for (int i = 0; i < 16; i++) {
                w[i] = uint32_t(msg[chunk + i * 4]) | (uint32_t(msg[chunk + i * 4 + 1]) << 8) | (uint32_t(msg[chunk + i * 4 + 2]) << 16) | (uint32_t(msg[chunk + i * 4 + 3]) << 24);
            }

            uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
            for (int i = 0; i < 64; i++) {
                uint32_t f, g;
                if (i < 16) { f = (b & c) | (~b & d); g = i; }
                else if (i < 32) { f = (d & b) | (~d & c); g = (5 * i + 1) % 16; }
                else if (i < 48) { f = b ^ c ^ d; g = (3 * i + 5) % 16; }
                else { f = c ^ (b | ~d); g = (7 * i) % 16; }

                const uint32_t tmp = d;
                d = c;
                c = b;
                const uint32_t x = a + f + K[i] + w[g];
                b = b + ((x << S[i]) | (x >> (32 - S[i])));
                a = tmp;
            }

            h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        }

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                digest[i * 4 + j] = uint8_t(h[i] >> (8 * j));
            }
        }
    }

    // Jabo Direct3D8 texture pack key: MD5 of the texture decoded to A1R5G5B5 (little-endian, row by row, at the
    // tile's width and height), as 32 lowercase hex digits. Determined by matching textures dumped from the game
    // against a Jabo pack: CI4/CI8 with RGBA16 palettes and IA4 confirmed; the other formats follow the same rule.
    static std::string computeJaboKey(const State *state, const LoadTile &drawTile, uint16_t width, uint16_t height, uint32_t tlut) {
        if ((width == 0) || (height == 0) || (width > 1024) || (height > 1024)) {
            return std::string();
        }

        const uint32_t RDRAMBytes = RDRAMSize + 1;
        const LoadOperation &loadOp = state->rdp->rice.lastLoadOpByTMEM[drawTile.tmem];
        const uint8_t *RDRAM = reinterpret_cast<const uint8_t *>(state->RDRAM);
        auto readByte = [&](uint32_t address, bool &ok) -> uint8_t {
            if (address >= RDRAMBytes) {
                ok = false;
                return 0;
            }

            return RDRAM[address ^ 3];
        };

        const uint32_t bytesPerRow = loadOp.texture.width << loadOp.texture.siz >> 1;
        const uint32_t startOffset = (loadOp.tile.uls >> 2) << loadOp.texture.siz >> 1;
        uint32_t start, rowStride;
        if (loadOp.type == LoadOperation::Type::Block) {
            start = loadOp.texture.address + startOffset + bytesPerRow * loadOp.tile.ult;
            rowStride = uint32_t(drawTile.line) << 3;
        }
        else if (loadOp.type == LoadOperation::Type::Tile) {
            start = loadOp.texture.address + startOffset + bytesPerRow * (loadOp.tile.ult >> 2);
            rowStride = bytesPerRow;
        }
        else {
            return std::string();
        }

        // Palette (CI): the RGBA16 entries the texture's palette bank points at.
        uint32_t paletteStart = 0;
        if (drawTile.fmt == G_IM_FMT_CI) {
            const bool CI4 = (drawTile.siz == G_IM_SIZ_4b);
            const int32_t paletteBase = RDP_TMEM_WORDS >> 1;
            const int32_t bankOffset = CI4 ? (drawTile.palette << 4) : 0;
            const LoadOperation *paletteLoadOp = &state->rdp->rice.lastLoadOpByTMEM[paletteBase + bankOffset];
            uint32_t extra = 0;
            if ((paletteLoadOp->texture.address == 0) && (bankOffset != 0)) {
                // The whole palette was loaded at once: index into it by bank.
                paletteLoadOp = &state->rdp->rice.lastLoadOpByTMEM[paletteBase];
                extra = uint32_t(bankOffset) * 8;
            }

            const uint32_t paletteBytesPerRow = paletteLoadOp->texture.width << paletteLoadOp->texture.siz >> 1;
            paletteStart = paletteLoadOp->texture.address + ((paletteLoadOp->tile.uls >> 2) << paletteLoadOp->texture.siz >> 1) + paletteBytesPerRow * (paletteLoadOp->tile.ult >> 2) + extra;
        }

        bool ok = true;
        auto rgba16To1555 = [](uint16_t c) { return uint16_t(((c & 1) << 15) | (c >> 1)); };
        auto intensityTo1555 = [](uint32_t i8, bool opaque) { const uint16_t i5 = uint16_t(i8 >> 3); return uint16_t((opaque ? 0x8000 : 0) | (i5 << 10) | (i5 << 5) | i5); };
        std::vector<uint8_t> pixels;
        pixels.reserve(size_t(width) * height * 2);
        for (uint32_t y = 0; (y < height) && ok; y++) {
            const uint32_t row = start + y * rowStride;
            for (uint32_t x = 0; (x < width) && ok; x++) {
                uint16_t out = 0;
                switch (drawTile.siz) {
                case G_IM_SIZ_4b: {
                    const uint8_t b = readByte(row + (x >> 1), ok);
                    const uint32_t v = (x & 1) ? (b & 0xF) : (b >> 4);
                    if (drawTile.fmt == G_IM_FMT_CI) {
                        out = rgba16To1555(uint16_t((readByte(paletteStart + v * 2, ok) << 8) | readByte(paletteStart + v * 2 + 1, ok)));
                    }
                    else if (drawTile.fmt == G_IM_FMT_IA) {
                        out = intensityTo1555(((v >> 1) * 255) / 7, (v & 1) != 0);
                    }
                    else {
                        out = intensityTo1555(v * 17, true);
                    }
                    break;
                }
                case G_IM_SIZ_8b: {
                    const uint32_t v = readByte(row + x, ok);
                    if (drawTile.fmt == G_IM_FMT_CI) {
                        out = rgba16To1555(uint16_t((readByte(paletteStart + v * 2, ok) << 8) | readByte(paletteStart + v * 2 + 1, ok)));
                    }
                    else if (drawTile.fmt == G_IM_FMT_IA) {
                        out = intensityTo1555((v >> 4) * 17, (v & 0xF) >= 8);
                    }
                    else {
                        out = intensityTo1555(v, true);
                    }
                    break;
                }
                case G_IM_SIZ_16b: {
                    const uint16_t c = uint16_t((readByte(row + x * 2, ok) << 8) | readByte(row + x * 2 + 1, ok));
                    out = (drawTile.fmt == G_IM_FMT_IA) ? intensityTo1555(c >> 8, (c & 0xFF) >= 0x80) : rgba16To1555(c);
                    break;
                }
                default:
                    return std::string();
                }

                pixels.push_back(uint8_t(out));
                pixels.push_back(uint8_t(out >> 8));
            }
        }

        if (!ok) {
            return std::string();
        }

        uint8_t digest[16];
        md5(pixels.data(), pixels.size(), digest);
        char hex[33];
        for (int i = 0; i < 16; i++) {
            snprintf(hex + i * 2, 3, "%02x", digest[i]);
        }

        return std::string(hex, 32);
    }

    // TextureManager

    void TextureManager::uploadEmpty(State *state, TextureCache *textureCache, uint64_t creationFrame, uint16_t width, uint16_t height, uint64_t replacementHash) {
        if (hashSet.find(replacementHash) == hashSet.end()) {
            hashSet.insert(replacementHash);
            textureCache->queueGPUUploadTMEM(replacementHash, creationFrame, nullptr, 0, width, height, 0, LoadTile(), false);
        }
    }

    uint64_t TextureManager::uploadTMEM(State *state, const LoadTile &loadTile, TextureCache *textureCache, uint64_t creationFrame, uint16_t byteOffset, uint16_t byteCount, uint16_t width, uint16_t height, uint32_t tlut) {
        XXH3_state_t xxh3;
        XXH3_64bits_reset(&xxh3);
        const uint8_t *TMEM = reinterpret_cast<const uint8_t *>(state->rdp->TMEM);
        XXH3_64bits_update(&xxh3, &TMEM[byteOffset], byteCount);
        XXH3_64bits_update(&xxh3, &byteOffset, sizeof(byteOffset));
        XXH3_64bits_update(&xxh3, &byteCount, sizeof(byteCount));
        const uint64_t hash = XXH3_64bits_digest(&xxh3);
        if (hashSet.find(hash) == hashSet.end()) {
            hashSet.insert(hash);
            textureCache->queueGPUUploadTMEM(hash, creationFrame, TMEM, RDP_TMEM_BYTES, width, height, 0, LoadTile(), false);
        }
        
        // Dump memory contents into a file if the process is active.
        if (!state->dumpingTexturesDirectory.empty()) {
            // Since width and height are not exactly guaranteed to be sane values when using raw TMEM, ensure we only dump textures when the values make sense.
            const bool validTextureCheck = (width > 0x0) && (height > 0x0);
            const bool bigTextureCheck = (width > 0x1000) || (height > 0x1000);
            if (validTextureCheck && !bigTextureCheck) {
                dumpTexture(hash, state, loadTile, width, height, tlut);
            }
        }

        return hash;
    }

    uint64_t TextureManager::uploadTexture(State *state, const LoadTile &loadTile, TextureCache *textureCache, uint64_t creationFrame, uint16_t width, uint16_t height, uint32_t tlut) {
        const uint8_t *TMEM = reinterpret_cast<const uint8_t *>(state->rdp->TMEM);
        uint64_t hash = TMEMHasher::hash(TMEM, loadTile, width, height, tlut, TMEMHasher::CurrentHashVersion);
        if (hashSet.find(hash) == hashSet.end()) {
            hashSet.insert(hash);
            // Keys used by Rice/GLideN64 and Jabo texture packs, for packs without an rt64.json mapping.
            const std::string legacyKeys = computeRiceKey(state, loadTile, width, height, tlut) + "|" + computeJaboKey(state, loadTile, width, height, tlut);
            textureCache->queueGPUUploadTMEM(hash, creationFrame, TMEM, RDP_TMEM_BYTES, width, height, tlut, loadTile, true, legacyKeys);
        }

        // Dump memory contents into a file if the process is active.
        if (!state->dumpingTexturesDirectory.empty()) {
            dumpTexture(hash, state, loadTile, width, height, tlut);
        }

        return hash;
    }

    void TextureManager::dumpTexture(uint64_t hash, State *state, const LoadTile &loadTile, uint16_t width, uint16_t height, uint32_t tlut) {
        if (dumpedSet.find(hash) != dumpedSet.end()) {
            return;
        }

        // Insert into set regardless of whether the dump is successful or not.
        dumpedSet.insert(hash);
        
        // Dump the entirety of TMEM.
        char baseName[64];
        snprintf(baseName, sizeof(baseName), "%016" PRIx64 ".v%u", hash, TMEMHasher::CurrentHashVersion);
        std::filesystem::path dumpTmemPath = state->dumpingTexturesDirectory / (std::string(baseName) + ".tmem");
        std::ofstream dumpTmemStream(dumpTmemPath, std::ios::binary);
        if (dumpTmemStream.is_open()) {
            const char *TMEM = reinterpret_cast<const char *>(state->rdp->TMEM);
            dumpTmemStream.write(TMEM, RDP_TMEM_BYTES);
            dumpTmemStream.close();
        }

        // Dump the RDRAM last loaded into the TMEM address pointed to by the tile. Required for generating hashes used by Rice.
        const LoadOperation &loadOp = state->rdp->rice.lastLoadOpByTMEM[loadTile.tmem];
        uint32_t rdramStart = loadOp.texture.address;
        uint32_t rdramCount = 0;
        uint32_t commonBytesOffset = (loadOp.tile.uls >> 2) << loadOp.texture.siz >> 1;
        uint32_t commonBytesPerRow = loadOp.texture.width << loadOp.texture.siz >> 1;
        if (loadOp.type == LoadOperation::Type::Block) {
            uint32_t wordCount = ((loadOp.tile.lrs - loadOp.tile.uls) >> (4 - loadOp.tile.siz)) + 1;
            rdramStart = loadOp.texture.address + commonBytesOffset + commonBytesPerRow * loadOp.tile.ult;
            rdramCount = (wordCount << 3);

            // Increase the amount of RDRAM dumped by textures that require padding when using load block.
            commonBytesPerRow = std::max(commonBytesPerRow, uint32_t(loadTile.line) << 3U);
        }
        else if (loadOp.type == LoadOperation::Type::Tile) {
            uint32_t rowCount = 1 + ((loadOp.tile.lrt >> 2) - (loadOp.tile.ult >> 2));
            uint32_t tileWidth = ((loadOp.tile.lrs >> 2) - (loadOp.tile.uls >> 2));
            uint32_t wordsPerRow = (tileWidth >> (4 - loadOp.tile.siz)) + 1;
            rdramStart = loadOp.texture.address + commonBytesOffset + commonBytesPerRow * (loadOp.tile.ult >> 2);
            rdramCount = rowCount * commonBytesPerRow;
        }
        
        // Dump more RDRAM if necessary if it doesn't cover what the tile could possibly sample.
        uint32_t loadTileBpr = width << loadTile.siz >> 1;
        rdramCount = std::max(rdramCount, std::max(loadTileBpr, commonBytesPerRow) * height);

        if (rdramCount > 0) {
            std::filesystem::path dumpRdramPath = state->dumpingTexturesDirectory / (std::string(baseName) + ".rice.rdram");
            std::ofstream dumpRdramStream(dumpRdramPath, std::ios::binary);
            if (dumpRdramStream.is_open()) {
                const char *RDRAM = reinterpret_cast<const char *>(state->RDRAM);
                dumpRdramStream.write(&RDRAM[rdramStart], rdramCount);
                dumpRdramStream.close();
            }

            std::filesystem::path dumpRdramInfoPath = state->dumpingTexturesDirectory / (std::string(baseName) + ".rice.json");
            std::ofstream dumpRdramInfoStream(dumpRdramInfoPath);
            if (dumpRdramInfoStream.is_open()) {
                json jroot;
                jroot["tile"] = loadOp.tile;
                jroot["type"] = loadOp.type;
                jroot["texture"] = loadOp.texture;
                dumpRdramInfoStream << std::setw(4) << jroot << std::endl;
                dumpRdramInfoStream.close();
            }
        }
        
        // Repeat a similar process for dumping the palette.
        if (tlut > 0) {
            const bool CI4 = (loadTile.siz == G_IM_SIZ_4b);
            const int32_t paletteTMEM = (RDP_TMEM_WORDS >> 1) + (CI4 ? (loadTile.palette << 4) : 0);
            const LoadOperation &paletteLoadOp = state->rdp->rice.lastLoadOpByTMEM[paletteTMEM];
            uint32_t paletteBytesOffset = (paletteLoadOp.tile.uls >> 2) << paletteLoadOp.texture.siz >> 1;
            uint32_t paletteBytesPerRow = paletteLoadOp.texture.width << paletteLoadOp.texture.siz >> 1;
            const uint32_t rowCount = 1 + ((paletteLoadOp.tile.lrt >> 2) - (paletteLoadOp.tile.ult >> 2));
            const uint32_t wordsPerRow = ((paletteLoadOp.tile.lrs >> 2) - (paletteLoadOp.tile.uls >> 2)) + 1;
            uint32_t paletteRdramStart = paletteLoadOp.texture.address + paletteBytesOffset + paletteBytesPerRow * (paletteLoadOp.tile.ult >> 2);
            uint32_t paletteRdramCount = (rowCount - 1) * paletteBytesPerRow + (wordsPerRow << 3);
            if (paletteRdramCount > 0) {
                std::filesystem::path dumpPaletteRdramPath = state->dumpingTexturesDirectory / (std::string(baseName) + ".rice.palette.rdram");
                std::ofstream dumpPaletteRdramStream(dumpPaletteRdramPath, std::ios::binary);
                if (dumpPaletteRdramStream.is_open()) {
                    const char *RDRAM = reinterpret_cast<const char *>(state->RDRAM);
                    dumpPaletteRdramStream.write(&RDRAM[paletteRdramStart], paletteRdramCount);
                    dumpPaletteRdramStream.close();
                }
            }

            std::filesystem::path dumpPaletteRdramInfoPath = state->dumpingTexturesDirectory / (std::string(baseName) + ".rice.palette.json");
            std::ofstream dumpPaletteRdramInfoStream(dumpPaletteRdramInfoPath);
            if (dumpPaletteRdramInfoStream.is_open()) {
                json jroot;
                jroot["tile"] = paletteLoadOp.tile;
                jroot["type"] = paletteLoadOp.type;
                jroot["texture"] = paletteLoadOp.texture;
                dumpPaletteRdramInfoStream << std::setw(4) << jroot << std::endl;
                dumpPaletteRdramInfoStream.close();
            }
        }

        // Dump the parameters of the tile into a JSON file.
        std::filesystem::path dumpTilePath = state->dumpingTexturesDirectory / (std::string(baseName) + ".tile.json");
        std::ofstream dumpTileStream(dumpTilePath);
        if (dumpTileStream.is_open()) {
            json jroot;
            jroot["tile"] = loadTile;
            jroot["width"] = width;
            jroot["height"] = height;

            // Serialize the TLUT into an enum instead.
            if (tlut == G_TT_RGBA16) {
                jroot["tlut"] = LoadTLUT::RGBA16;
            }
            else if (tlut == G_TT_IA16) {
                jroot["tlut"] = LoadTLUT::IA16;
            }
            else {
                jroot["tlut"] = LoadTLUT::None;
            }

            dumpTileStream << std::setw(4) << jroot << std::endl;
            dumpTileStream.close();
        }
    }

    void TextureManager::removeHashes(const std::vector<uint64_t> &hashes) {
        for (uint64_t hash : hashes) {
            hashSet.erase(hash);
        }
    }
};