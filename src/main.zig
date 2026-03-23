const std = @import("std");

const Operation = enum {
    init,
    add,
    sub,
    mul,
    sigmoid,
    pow,
};

const TrackedF32 = struct {
    val: f32,
    op: Operation,
    in1: ?*TrackedF32,
    in2: ?*TrackedF32,

    fn init(alloc: std.mem.Allocator, val: f32) !*TrackedF32 {
        const ret = try alloc.create(TrackedF32);
        ret.* = .{
            .val = val,
            .op = .init,
            .in1 = null,
            .in2 = null,
        };
        return ret;
    }

    fn add(alloc: std.mem.Allocator, a: *TrackedF32, b: *TrackedF32) !*TrackedF32 {
        const ret = try alloc.create(TrackedF32);
        ret.* = .{
            .val = a.val + b.val,
            .op = .add,
            .in1 = a,
            .in2 = b,
        };
        return ret;
    }

    fn sub(alloc: std.mem.Allocator, a: *TrackedF32, b: *TrackedF32) !*TrackedF32 {
        const ret = try alloc.create(TrackedF32);
        ret.* = .{
            .val = a.val - b.val,
            .op = .sub,
            .in1 = a,
            .in2 = b,
        };
        return ret;
    }

    fn mul(alloc: std.mem.Allocator, a: *TrackedF32, b: *TrackedF32) !*TrackedF32 {
        const ret = try alloc.create(TrackedF32);
        ret.* = .{
            .val = a.val * b.val,
            .op = .mul,
            .in1 = a,
            .in2 = b,
        };
        return ret;
    }

    fn sigmoid(alloc: std.mem.Allocator, a: *TrackedF32) !*TrackedF32 {
        const ret = try alloc.create(TrackedF32);
        ret.* = .{
            .val = 1.0 / (1.0 + std.math.exp(-a.val)),
            .op = .sigmoid,
            .in1 = a,
            .in2 = null,
        };
        return ret;
    }

    fn pow(alloc: std.mem.Allocator, a: *TrackedF32, b: *TrackedF32) !*TrackedF32 {
        const ret = try alloc.create(TrackedF32);
        ret.* = .{
            .val = std.math.pow(f32, a.val, b.val),
            .op = .pow,
            .in1 = a,
            .in2 = b,
        };
        return ret;
    }
};

fn sigmoid(in: f32) f32 {
    return 1.0 / 1.0 + std.math.exp(-in);
}

const Network = struct {
    weights: [2]*TrackedF32,
    biases: [2]*TrackedF32,

    fn init(alloc: std.mem.Allocator) !Network {
        return .{
            .weights = .{
                try TrackedF32.init(alloc, 0),
                try TrackedF32.init(alloc, 0),
            },
            .biases = .{
                try TrackedF32.init(alloc, 0),
                try TrackedF32.init(alloc, 0),
            },
        };
    }

    fn run(self: *Network, alloc: std.mem.Allocator, a: f32, b: f32) !*TrackedF32 {
        const a_tracked = try TrackedF32.init(alloc, a);
        const b_tracked = try TrackedF32.init(alloc, b);

        const a_out = try TrackedF32.add(alloc, try TrackedF32.mul(alloc, self.weights[0], a_tracked), self.biases[0]);
        const b_out = try TrackedF32.add(alloc, try TrackedF32.mul(alloc, self.weights[1], b_tracked), self.biases[1]);

        return TrackedF32.sigmoid(alloc, try TrackedF32.add(alloc, a_out, b_out));
    }
};

fn printNet(input: *TrackedF32, indent_level: usize) void {
    for (0..indent_level) |_| {
        std.debug.print("\t", .{});
    }
    std.debug.print("{s} : {d}\n", .{ @tagName(input.op), input.val });

    if (input.in1) |in1| {
        printNet(in1, indent_level + 1);
    }

    if (input.in1) |in2| {
        printNet(in2, indent_level + 1);
    }
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    var net = try Network.init(arena.allocator());
    const output = try net.run(arena.allocator(), 3, 10);

    const expected = try TrackedF32.init(arena.allocator(), 0);
    const loss = try TrackedF32.pow(arena.allocator(), try TrackedF32.sub(arena.allocator(), output, expected), try TrackedF32.init(arena.allocator(), 2));
    printNet(loss, 0);
    std.debug.print("{d}", .{output.val});
}
