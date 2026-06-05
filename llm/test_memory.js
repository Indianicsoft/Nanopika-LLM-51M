// Simple test for MemoryArena (run in node or browser)
// Since memory.js doesn't export, we'll wrap it or just use it as is if we can.
// For testing purposes, we'll simulate the environment.

const fs = require('fs');
const code = fs.readFileSync('./memory.js', 'utf8');

// Use eval to run the code in this scope to access the local variables
eval(code);

console.log("MemoryArena instantiated with capacity:", arena.capacity);
console.log("Allocated vecQ size:", vecQ.length);
console.log("Current arena offset:", arena.offset);

// Check if views share the same buffer
if (vecQ.buffer === arena.buffer && vecK.buffer === arena.buffer) {
    console.log("SUCCESS: Tensors share the same ArrayBuffer.");
} else {
    console.log("FAILURE: Tensors do not share the same buffer.");
}

// Verify sequence of offsets
const expectedOffset = (D_HEAD * 3 + SEQ_LEN * 2 + D_FF + D_MODEL * 3) * 4;
if (arena.offset === expectedOffset) {
    console.log("SUCCESS: Arena offset matches expected value.");
} else {
    console.log(`FAILURE: Arena offset mismatch. Expected ${expectedOffset}, got ${arena.offset}`);
}

// Test reset
arena.reset();
if (arena.offset === 0) {
    console.log("SUCCESS: Arena reset worked.");
} else {
    console.log("FAILURE: Arena reset failed.");
}

// Test overflow
try {
    arena.allocate(arena.capacity + 1);
    console.log("FAILURE: Arena did not throw on overflow.");
} catch (e) {
    if (e.message.includes("MemoryArena overflow")) {
        console.log("SUCCESS: Arena threw correct overflow error.");
    } else {
        console.log("FAILURE: Arena threw wrong error:", e.message);
    }
}
