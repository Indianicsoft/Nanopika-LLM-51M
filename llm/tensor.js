/**
 * tensor.js - A standalone, high-performance Tensor math library with Autograd and Adam Optimizer.
 * Built for 15M parameter Transformer training in vanilla JavaScript.
 * 
 * Constraints:
 * - Zero external libraries.
 * - Float32Array contiguous memory.
 * - Built-in computation graph for backpropagation.
 */

/**
 * Utility to calculate strides for a given shape.
 * @param {number[]} shape 
 * @returns {number[]}
 */
function calculateStrides(shape) {
    const strides = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

export class Tensor {
    /**
     * @param {Float32Array|number[]} data 
     * @param {object} options 
     */
    constructor(data, { shape = null, requiresGrad = false, prev = [], op = "" } = {}) {
        if (data instanceof Float32Array) {
            this.data = data;
        } else {
            this.data = new Float32Array(data);
        }

        this.shape = shape || [this.data.length];
        this.strides = calculateStrides(this.shape);
        this.requiresGrad = requiresGrad;
        this.grad = requiresGrad ? new Float32Array(this.data.length) : null;
        
        // Autograd attributes
        this.prev = new Set(prev);
        this.op = op;
        this._backward = () => {};
    }

    /**
     * Reset gradients to zero.
     */
    zeroGrad() {
        if (this.grad) {
            this.grad.fill(0);
        }
    }

    /**
     * Executes the global backward pass starting from this tensor.
     */
    backward() {
        // Build topological order
        const topo = [];
        const visited = new Set();
        const buildTopo = (v) => {
            if (!visited.has(v)) {
                visited.add(v);
                for (let child of v.prev) {
                    buildTopo(child);
                }
                topo.push(v);
            }
        };
        buildTopo(this);

        // Seed the gradient of the root
        if (this.grad) {
            this.grad.fill(1.0);
        }

        // Backpropagate in reverse topological order
        for (let i = topo.length - 1; i >= 0; i--) {
            topo[i]._backward();
        }
    }

    /**
     * Element-wise addition: out = this + other
     * @param {Tensor} other 
     * @returns {Tensor}
     */
    add(other) {
        if (this.data.length !== other.data.length) {
            throw new Error(`Shape mismatch for add: [${this.shape}] and [${other.shape}]`);
        }

        const out = new Tensor(new Float32Array(this.data.length), {
            shape: this.shape,
            requiresGrad: this.requiresGrad || other.requiresGrad,
            prev: [this, other],
            op: "add"
        });

        // Forward pass
        for (let i = 0; i < this.data.length; i++) {
            out.data[i] = this.data[i] + other.data[i];
        }

        // Backward pass
        out._backward = () => {
            if (this.requiresGrad) {
                for (let i = 0; i < this.data.length; i++) {
                    this.grad[i] += out.grad[i];
                }
            }
            if (other.requiresGrad) {
                for (let i = 0; i < other.data.length; i++) {
                    other.grad[i] += out.grad[i];
                }
            }
        };

        return out;
    }

    /**
     * 2D Matrix Multiplication: out = this @ other
     * Optimized using row-major sequential access (i, k, j loop) for CPU cache locality.
     * @param {Tensor} other 
     * @returns {Tensor}
     */
    matmul(other) {
        const [M, K] = this.shape;
        const [K2, N] = other.shape;

        if (K !== K2) {
            throw new Error(`Matrix multiplication inner dimensions must match: ${K} and ${K2}`);
        }

        const out = new Tensor(new Float32Array(M * N), {
            shape: [M, N],
            requiresGrad: this.requiresGrad || other.requiresGrad,
            prev: [this, other],
            op: "matmul"
        });

        const A = this.data;
        const B = other.data;
        const C = out.data;

        // Forward Pass: Optimized i, k, j loop
        for (let i = 0; i < M; i++) {
            const iOffset = i * K;
            const outOffset = i * N;
            for (let k = 0; k < K; k++) {
                const aik = A[iOffset + k];
                const kOffsetB = k * N;
                for (let j = 0; j < N; j++) {
                    C[outOffset + j] += aik * B[kOffsetB + j];
                }
            }
        }

        // Backward Pass
        out._backward = () => {
            const dC = out.grad;
            if (this.requiresGrad) {
                const dA = this.grad;
                // dA = dC @ B.T
                // [M, N] @ [N, K] -> [M, K]
                for (let i = 0; i < M; i++) {
                    const iOffsetDA = i * K;
                    const iOffsetDC = i * N;
                    for (let j = 0; j < N; j++) {
                        const dcij = dC[iOffsetDC + j];
                        const jOffsetB = j * K; // Transposed B indexing
                        for (let k = 0; k < K; k++) {
                            // B.T[j, k] is B[k, j]
                            dA[iOffsetDA + k] += dcij * B[k * N + j];
                        }
                    }
                }
            }
            if (other.requiresGrad) {
                const dB = other.grad;
                // dB = A.T @ dC
                // [K, M] @ [M, N] -> [K, N]
                for (let k = 0; k < K; k++) {
                    const kOffsetDB = k * N;
                    for (let i = 0; i < M; i++) {
                        const aik = A[i * K + k]; // A.T[k, i] is A[i, k]
                        const iOffsetDC = i * N;
                        for (let j = 0; j < N; j++) {
                            dB[kOffsetDB + j] += aik * dC[iOffsetDC + j];
                        }
                    }
                }
            }
        };

        return out;
    }

    /**
     * Softmax along a specific axis (currently only last axis supported for simplicity).
     * Includes numerical stability by subtracting max value.
     * @param {number} axis 
     * @returns {Tensor}
     */
    softmax(axis = -1) {
        if (axis !== -1 && axis !== this.shape.length - 1) {
            throw new Error("Currently only last-axis softmax is implemented.");
        }

        const out = new Tensor(new Float32Array(this.data.length), {
            shape: this.shape,
            requiresGrad: this.requiresGrad,
            prev: [this],
            op: "softmax"
        });

        const rows = this.data.length / this.shape[this.shape.length - 1];
        const cols = this.shape[this.shape.length - 1];

        for (let r = 0; r < rows; r++) {
            const offset = r * cols;
            
            // 1. Find max for numerical stability
            let maxVal = -Infinity;
            for (let c = 0; c < cols; c++) {
                if (this.data[offset + c] > maxVal) maxVal = this.data[offset + c];
            }

            // 2. Compute exp and sum
            let sumExp = 0;
            for (let c = 0; c < cols; c++) {
                const expVal = Math.exp(this.data[offset + c] - maxVal);
                out.data[offset + c] = expVal;
                sumExp += expVal;
            }

            // 3. Normalize
            for (let c = 0; c < cols; c++) {
                out.data[offset + c] /= sumExp;
            }
        }

        // Backward Pass: dL/dx_i = p_i * (dL/dp_i - sum_j(p_j * dL/dp_j))
        out._backward = () => {
            if (this.requiresGrad) {
                for (let r = 0; r < rows; r++) {
                    const offset = r * cols;
                    
                    let sumPDG = 0;
                    for (let c = 0; c < cols; c++) {
                        sumPDG += out.data[offset + c] * out.grad[offset + c];
                    }

                    for (let c = 0; c < cols; c++) {
                        const pi = out.data[offset + c];
                        this.grad[offset + c] += pi * (out.grad[offset + c] - sumPDG);
                    }
                }
            }
        };

        return out;
    }

    /**
     * Layer Normalization: out = (x - mean) / sqrt(var + eps) * gamma + beta
     * @param {Tensor} gamma 
     * @param {Tensor} beta 
     * @param {number} eps 
     * @returns {Tensor}
     */
    layerNorm(gamma, beta, eps = 1e-5) {
        const lastDim = this.shape[this.shape.length - 1];
        const rows = this.data.length / lastDim;

        const out = new Tensor(new Float32Array(this.data.length), {
            shape: this.shape,
            requiresGrad: this.requiresGrad || gamma.requiresGrad || beta.requiresGrad,
            prev: [this, gamma, beta],
            op: "layerNorm"
        });

        // Storage for intermediate values needed for backward pass
        const means = new Float32Array(rows);
        const vars = new Float32Array(rows);

        for (let r = 0; r < rows; r++) {
            const offset = r * lastDim;
            
            // 1. Mean
            let mean = 0;
            for (let c = 0; c < lastDim; c++) mean += this.data[offset + c];
            mean /= lastDim;
            means[r] = mean;

            // 2. Variance
            let variance = 0;
            for (let c = 0; c < lastDim; c++) {
                const diff = this.data[offset + c] - mean;
                variance += diff * diff;
            }
            variance /= lastDim;
            vars[r] = variance;

            // 3. Normalize & Scale/Shift
            const invStd = 1.0 / Math.sqrt(variance + eps);
            for (let c = 0; c < lastDim; c++) {
                const xHat = (this.data[offset + c] - mean) * invStd;
                out.data[offset + c] = xHat * gamma.data[c] + beta.data[c];
            }
        }

        out._backward = () => {
            if (this.requiresGrad) {
                for (let r = 0; r < rows; r++) {
                    const offset = r * lastDim;
                    const mean = means[r];
                    const variance = vars[r];
                    const invStd = 1.0 / Math.sqrt(variance + eps);

                    let dLdxhatSum = 0;
                    let dLdxhatXdiffSum = 0;

                    for (let c = 0; c < lastDim; c++) {
                        const xHat = (this.data[offset + c] - mean) * invStd;
                        const dLdxhat = out.grad[offset + c] * gamma.data[c];
                        dLdxhatSum += dLdxhat;
                        dLdxhatXdiffSum += dLdxhat * (this.data[offset + c] - mean);
                        
                        // Gradients for gamma and beta (if they require it)
                        if (gamma.requiresGrad) gamma.grad[c] += out.grad[offset + c] * xHat;
                        if (beta.requiresGrad) beta.grad[c] += out.grad[offset + c];
                    }

                    // Gradients for input x
                    for (let c = 0; c < lastDim; c++) {
                        const xDiff = this.data[offset + c] - mean;
                        const term1 = lastDim * out.grad[offset + c] * gamma.data[c];
                        const term2 = dLdxhatSum;
                        const term3 = xDiff * invStd * invStd * dLdxhatXdiffSum;
                        this.grad[offset + c] += (invStd / lastDim) * (term1 - term2 - term3);
                    }
                }
            }
        };

        return out;
    }
}

/**
 * Adam Optimizer
 */
export class AdamOptimizer {
    /**
     * @param {Tensor[]} params 
     * @param {number} lr 
     * @param {number} beta1 
     * @param {number} beta2 
     * @param {number} eps 
     */
    constructor(params, lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
        this.params = params.filter(p => p.requiresGrad);
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.t = 0;

        // Initialize momentums
        this.m = this.params.map(p => new Float32Array(p.data.length));
        this.v = this.params.map(p => new Float32Array(p.data.length));
    }

    /**
     * Perform one optimization step.
     */
    step() {
        this.t++;
        const bc1 = 1 - Math.pow(this.beta1, this.t);
        const bc2 = 1 - Math.pow(this.beta2, this.t);

        for (let i = 0; i < this.params.length; i++) {
            const p = this.params[i];
            const m = this.m[i];
            const v = this.v[i];
            const g = p.grad;
            const data = p.data;

            for (let j = 0; j < data.length; j++) {
                // Update moments
                m[j] = this.beta1 * m[j] + (1 - this.beta1) * g[j];
                v[j] = this.beta2 * v[j] + (1 - this.beta2) * g[j] * g[j];

                // Bias correction
                const mHat = m[j] / bc1;
                const vHat = v[j] / bc2;

                // Update parameters
                data[j] -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
            }
        }
    }

    /**
     * Zero all gradients.
     */
    zeroGrad() {
        for (const p of this.params) {
            p.zeroGrad();
        }
    }
}
