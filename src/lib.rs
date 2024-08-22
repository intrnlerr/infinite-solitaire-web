mod cards;
mod utils;

use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
};

use cards::CardStack;
use wasm_bindgen::prelude::*;
use web_sys::{
    WebGl2RenderingContext, WebGlBuffer, WebGlProgram, WebGlShader, WebGlTexture,
    WebGlUniformLocation,
};

const CARD_WIDTH: i64 = 80;
const CARD_HEIGHT: i64 = 120;
/// the margin (in pixels) on the right of each column.
const COLUMN_MARGIN: i64 = 1;

thread_local! {
    static IS_DRAG: Cell<bool> = Cell::new(false);
    static CAMERA_X: Cell<i64> = Cell::new(0);
    static CAMERA_Y: Cell<i64> = Cell::new(-(CARD_HEIGHT + 2));

    static CANVAS_W: Cell<u32> = Cell::new(800);
    static CANVAS_H: Cell<u32> = Cell::new(600);

    static RENDER_CONTEXT: RefCell<Option<RenderInfo>> = RefCell::new(None);
    static COLUMNS: RefCell<Vec<cards::Column>> = RefCell::new(Vec::new());
    static TABLEAU: RefCell<HashMap<u32, cards::BitCard>> = RefCell::new(HashMap::new());

    static HELD_HAND: RefCell<cards::CardStack> = RefCell::new(CardStack::empty());
    static HELD_COLUMN: Cell<usize> = Cell::new(0);

    static CARD_DRAW_DATA: RefCell<Vec<i32>> = RefCell::new(Vec::new());

    static RNG: RefCell<oorandom::Rand32> = RefCell::new(oorandom::Rand32::new(0));

    static DRAG_START: Cell<[u32; 2]> = Cell::new([0, 0]);
    static MOUSE_POS: Cell<[u32; 2]> = Cell::new([0, 0]);
}

fn get_visible_columns(
    camera_x: i64,
    canvas_w: u32,
    columns: &mut Vec<cards::Column>,
) -> &[cards::Column] {
    let start = (camera_x / (CARD_WIDTH + COLUMN_MARGIN)).max(0) as usize;
    let end = ((camera_x + i64::from(canvas_w)) / (CARD_WIDTH + COLUMN_MARGIN) + 1).max(0) as usize;
    if columns.len() < end {
        columns.reserve(end - columns.len());
        for n in columns.len()..end {
            RNG.with_borrow_mut(|rng| columns.push(cards::Column::new(n as u32, rng)))
        }
    }
    &columns[start..end]
}

#[wasm_bindgen]
pub fn start() {
    if let Err(err) = _start() {
        wasm_bindgen::throw_val(err)
    }
}

fn _start() -> Result<(), JsValue> {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document.get_element_by_id("canvas").unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into::<web_sys::HtmlCanvasElement>()?;

    let context = canvas
        .get_context("webgl2")?
        .unwrap()
        .dyn_into::<WebGl2RenderingContext>()?;

    let vert_shader = compile_shader(
        &context,
        WebGl2RenderingContext::VERTEX_SHADER,
        r##"#version 300 es
 
        in vec4 position;
        in ivec2 card_pos;
        
        in int card_type;
        flat out int card_type_frag;

        in vec2 uv;
        out vec2 uv_f;

        uniform ivec2 camera;
        uniform uvec2 canvas_size;

        void main() {
            gl_Position = (
                ((position / vec4((vec2(canvas_size) * 0.5), 1.0, 1.0))
                + vec4((vec2(card_pos - camera) / (vec2(canvas_size) * 0.5)), 0.0, 0.0))
                * vec4(1.0, -1.0, 1.0, 1.0)
                + vec4(-1.0, 1.0, 0.0, 0.0)
            );
            card_type_frag = card_type;
            uv_f = uv;
        }
        "##,
    )?;

    let frag_shader = compile_shader(
        &context,
        WebGl2RenderingContext::FRAGMENT_SHADER,
        r##"#version 300 es
    
        precision highp float;
        flat in int card_type_frag;
        in vec2 uv_f;
        out vec4 outColor;

        uniform sampler2D tex;
        
        void main() {
            outColor = texture(tex, (uv_f + vec2((card_type_frag >> 2) & 15, 
            (card_type_frag & 3) + (card_type_frag >> 7))) / vec2(13.0, 5.0));
        }
        "##,
    )?;
    let program = link_program(&context, &vert_shader, &frag_shader)?;

    let camera_uniform = context
        .get_uniform_location(&program, "camera")
        .expect_throw("camera uniform must exist");
    let canvas_size_uniform = context
        .get_uniform_location(&program, "canvas_size")
        .expect_throw("canvas_size uniform must exist");
    let tex_uniform = context
        .get_uniform_location(&program, "tex")
        .expect_throw("tex uniform must exist");

    context.use_program(Some(&program));

    let card_texture = context
        .create_texture()
        .expect_throw("need to make texture");

    context.uniform1i(Some(&tex_uniform), 0);

    context.active_texture(WebGl2RenderingContext::TEXTURE0);
    context.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&card_texture));

    context.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
        WebGl2RenderingContext::TEXTURE_2D,
        0,
        WebGl2RenderingContext::RGBA as i32,
        1,
        1,
        0,
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::UNSIGNED_BYTE,
        Some(&[255, 255, 255, 255]),
    )?;

    let vertices: [f32; 8] = [
        0.0,
        0.0,
        CARD_WIDTH as f32,
        0.0,
        0.0,
        CARD_HEIGHT as f32,
        CARD_WIDTH as f32,
        CARD_HEIGHT as f32,
    ];

    let position_attribute_location = context.get_attrib_location(&program, "position");
    let card_data_attrib = context.get_attrib_location(&program, "card_pos");
    let card_type_attrib = context.get_attrib_location(&program, "card_type");
    let uv_attrib = context.get_attrib_location(&program, "uv");
    println!("{position_attribute_location}");

    let buffer = context.create_buffer().ok_or("Failed to create buffer")?;
    context.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&buffer));

    // Note that `Float32Array::view` is somewhat dangerous (hence the
    // `unsafe`!). This is creating a raw view into our module's
    // `WebAssembly.Memory` buffer, but if we allocate more pages for ourself
    // (aka do a memory allocation in Rust) it'll cause the buffer to change,
    // causing the `Float32Array` to be invalid.
    //
    // As a result, after `Float32Array::view` we have to be very careful not to
    // do any memory allocations before it's dropped.
    unsafe {
        let positions_array_buf_view = js_sys::Float32Array::view(&vertices);

        context.buffer_data_with_array_buffer_view(
            WebGl2RenderingContext::ARRAY_BUFFER,
            &positions_array_buf_view,
            WebGl2RenderingContext::STATIC_DRAW,
        );
    }

    let vao = context
        .create_vertex_array()
        .ok_or("Could not create vertex array object")?;
    context.bind_vertex_array(Some(&vao));

    context.vertex_attrib_pointer_with_i32(
        position_attribute_location as u32,
        2,
        WebGl2RenderingContext::FLOAT,
        false,
        0,
        0,
    );

    context.enable_vertex_attrib_array(position_attribute_location as u32);
    context.vertex_attrib_divisor(position_attribute_location as u32, 0);

    let uv_buffer = context
        .create_buffer()
        .expect_throw("couldn't make uv buffer!");

    context.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&uv_buffer));

    unsafe {
        let uv = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let view = js_sys::Float32Array::view(&uv);

        context.buffer_data_with_array_buffer_view(
            WebGl2RenderingContext::ARRAY_BUFFER,
            &view,
            WebGl2RenderingContext::STATIC_DRAW,
        );
    }

    context.vertex_attrib_pointer_with_i32(
        uv_attrib as u32,
        2,
        WebGl2RenderingContext::FLOAT,
        false,
        0,
        0,
    );
    context.enable_vertex_attrib_array(uv_attrib as u32);

    let card_data_buffer = context
        .create_buffer()
        .expect_throw("couldn't make card buffer!");

    context.bind_buffer(
        WebGl2RenderingContext::ARRAY_BUFFER,
        Some(&card_data_buffer),
    );

    {
        let instances = [200, 200, 0];
        unsafe {
            let view = js_sys::Uint32Array::view(&instances);

            context.buffer_data_with_array_buffer_view(
                WebGl2RenderingContext::ARRAY_BUFFER,
                &view,
                WebGl2RenderingContext::DYNAMIC_DRAW,
            );
        }
    }

    context.vertex_attrib_i_pointer_with_i32(
        card_data_attrib as u32,
        2,
        WebGl2RenderingContext::INT,
        (size_of::<i32>() * 3) as i32,
        0,
    );

    context.vertex_attrib_i_pointer_with_i32(
        card_type_attrib as u32,
        1,
        WebGl2RenderingContext::INT,
        (size_of::<i32>() * 3) as i32,
        (size_of::<i32>() * 2) as i32,
    );

    context.enable_vertex_attrib_array(card_data_attrib as u32);
    context.enable_vertex_attrib_array(card_type_attrib as u32);
    context.vertex_attrib_divisor(card_data_attrib as u32, 1);
    context.vertex_attrib_divisor(card_type_attrib as u32, 1);

    context.bind_vertex_array(Some(&vao));

    let info = RenderInfo {
        context,
        camera_uniform,
        card_data_buffer,
        canvas_size_uniform,
        card_texture,
    };
    RENDER_CONTEXT.set(Some(info));
    Ok(())
}

struct RenderInfo {
    context: WebGl2RenderingContext,
    camera_uniform: WebGlUniformLocation,
    canvas_size_uniform: WebGlUniformLocation,
    card_data_buffer: WebGlBuffer,
    card_texture: WebGlTexture,
}

fn draw(info: &RenderInfo) {
    let context = &info.context;
    context.uniform2i(
        Some(&info.camera_uniform),
        (CAMERA_X.get() % (CARD_WIDTH + COLUMN_MARGIN)) as i32,
        CAMERA_Y.get() as i32,
    );

    context.clear_color(0.0, 0.0, 0.0, 1.0);
    context.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT);

    let instances = CARD_DRAW_DATA.with_borrow_mut(|draw_data| {
        COLUMNS.with_borrow_mut(|columns| {
            setup_draw_data(draw_data, CAMERA_X.get(), CANVAS_W.get(), columns)
        });

        context.bind_buffer(
            WebGl2RenderingContext::ARRAY_BUFFER,
            Some(&info.card_data_buffer),
        );
        unsafe {
            let view = js_sys::Int32Array::view(&draw_data);

            context.buffer_data_with_array_buffer_view(
                WebGl2RenderingContext::ARRAY_BUFFER,
                &view,
                WebGl2RenderingContext::STATIC_DRAW,
            );
        }
        (draw_data.len() / 3) as i32
    });

    context.draw_arrays_instanced(WebGl2RenderingContext::TRIANGLE_STRIP, 0, 4, instances);
}

pub fn compile_shader(
    context: &WebGl2RenderingContext,
    shader_type: u32,
    source: &str,
) -> Result<WebGlShader, String> {
    let shader = context
        .create_shader(shader_type)
        .ok_or_else(|| String::from("Unable to create shader object"))?;
    context.shader_source(&shader, source);
    context.compile_shader(&shader);

    if context
        .get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(shader)
    } else {
        Err(context
            .get_shader_info_log(&shader)
            .unwrap_or_else(|| String::from("Unknown error creating shader")))
    }
}

pub fn link_program(
    context: &WebGl2RenderingContext,
    vert_shader: &WebGlShader,
    frag_shader: &WebGlShader,
) -> Result<WebGlProgram, String> {
    let program = context
        .create_program()
        .ok_or_else(|| String::from("Unable to create shader object"))?;

    context.attach_shader(&program, vert_shader);
    context.attach_shader(&program, frag_shader);
    context.link_program(&program);

    if context
        .get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(program)
    } else {
        Err(context
            .get_program_info_log(&program)
            .unwrap_or_else(|| String::from("Unknown error creating program object")))
    }
}

#[wasm_bindgen]
pub fn mouse_up(x: u32, y: u32) {
    IS_DRAG.set(false);

    // don't do card interactions if drag
    let [start_x, start_y] = DRAG_START.get();
    // taxicab should be good enough
    let dist = start_x.abs_diff(x) + start_y.abs_diff(y);
    if dist > 1 {
        return;
    }

    // check if hit card
    let mouse_pos = CAMERA_X.get() + x as i64;
    let col = mouse_pos / (CARD_WIDTH + COLUMN_MARGIN);
    let is_in_margin = mouse_pos % (CARD_WIDTH + COLUMN_MARGIN) >= CARD_WIDTH;
    let row = CAMERA_Y.get() + y as i64;
    if col >= 0 && !is_in_margin {
        if row >= 0 {
            let target_row = (row / (CARD_Y_MARGIN as i64)) as u32;
            HELD_HAND.with_borrow_mut(|hand| {
                COLUMNS.with_borrow_mut(|columns| {
                    let col = col as usize;
                    if hand.is_empty() {
                        let target = &mut columns[col];
                        let card_max_y = (target.under * CARD_Y_MARGIN as u32) as i64 + CARD_HEIGHT;
                        if target_row >= target.under && row < card_max_y {
                            let take = ((target_row - target.under) as usize)
                                .min(target.visible().len().saturating_sub(1) as usize);
                            hand.take_from(target.visible_mut(), take);
                            HELD_COLUMN.set(col);
                        }
                    } else if col == HELD_COLUMN.get() {
                        // return card to original spot
                        columns[col].append(hand);
                    } else {
                        let target_col = &mut columns[col];
                        let target_visible = target_col.visible_mut();
                        if target_visible.can_stack(hand.top()) {
                            target_visible.append(hand);
                            RNG.with_borrow_mut(|rng| {
                                columns[HELD_COLUMN.get()].maybe_reveal_card(rng)
                            })
                        }
                    }
                });
            });
        } else {
            HELD_HAND.with_borrow_mut(|hand| {
                if hand.len() == 1 {
                    let held_card = hand.top();
                    TABLEAU.with_borrow_mut(|tableau| match tableau.get_mut(&(col as u32)) {
                        Some(card)
                            if (held_card.is_next_card(*card)) && held_card.same_suit(*card) =>
                        {
                            *card = held_card;
                            hand.clear();
                            RNG.with_borrow_mut(|rng| {
                                COLUMNS.with_borrow_mut(|columns| {
                                    columns[HELD_COLUMN.get()].maybe_reveal_card(rng)
                                })
                            });
                        }
                        None if held_card.is_ace() => {
                            tableau.insert(col as u32, hand.top());
                            hand.clear();
                            RNG.with_borrow_mut(|rng| {
                                COLUMNS.with_borrow_mut(|columns| {
                                    columns[HELD_COLUMN.get()].maybe_reveal_card(rng)
                                })
                            });
                        }
                        _ => {}
                    })
                }
            })
        }

        RENDER_CONTEXT.with_borrow(|info| draw(info.as_ref().unwrap()))
    }
}

#[wasm_bindgen]
pub fn mouse_down(x: u32, y: u32) {
    IS_DRAG.set(true);
    DRAG_START.set([x, y]);
}

#[wasm_bindgen]
pub fn mouse_move(x: u32, y: u32) {
    if IS_DRAG.get() {
        let [prev_x, prev_y] = MOUSE_POS.get();
        let x_delta = x as i64 - prev_x as i64;
        let y_delta = y as i64 - prev_y as i64;
        CAMERA_X.set(CAMERA_X.get() - x_delta);
        CAMERA_Y.set(CAMERA_Y.get() - y_delta);

        MOUSE_POS.set([x, y]);

        // TODO: we only need to regenerate the draw data if the camera_x goes
        // TODO: past a multiple of (card_width + column_margin)
        RENDER_CONTEXT.with_borrow(|info: &Option<RenderInfo>| {
            draw(info.as_ref().expect_throw("render info must exist"))
        });
    } else {
        MOUSE_POS.set([x, y]);

        HELD_HAND.with_borrow(|hand| {
            if !hand.is_empty() {
                RENDER_CONTEXT
                    .with_borrow(|info: &Option<RenderInfo>| draw(info.as_ref().unwrap()));
            }
        })
    }
}

/// the vertical margin between cards
const CARD_Y_MARGIN: i32 = 24;

fn setup_draw_data(
    draw_data: &mut Vec<i32>,
    camera_x: i64,
    canvas_w: u32,
    columns: &mut Vec<cards::Column>,
) {
    fn append_stack(draw_data: &mut Vec<i32>, stack: &cards::CardStack, x: i32, y: i32) {
        for (card, offset) in stack.iter().zip(0..) {
            draw_data.push(x);
            draw_data.push(y + offset * CARD_Y_MARGIN);
            draw_data.push(card.as_u8() as i32);
        }
    }

    draw_data.clear();
    let leftmost_column = camera_x / (CARD_WIDTH + COLUMN_MARGIN);
    let first_offset = -(leftmost_column.min(0));
    for (column, i) in get_visible_columns(camera_x, canvas_w, columns)
        .iter()
        .zip(first_offset as i32..)
    {
        let x = i * (CARD_WIDTH + COLUMN_MARGIN) as i32;
        // TODO: clip the amount of cards to the camera height
        if column.is_visible_empty() && column.under == 0 {
            draw_data.push(x);
            draw_data.push(0);
            draw_data.push(0b10001111);
        } else {
            for n in 0..column.under {
                draw_data.push(x);
                draw_data.push(n as i32 * CARD_Y_MARGIN);
                draw_data.push(0b10001011);
            }
            append_stack(
                draw_data,
                column.visible(),
                x,
                column.under as i32 * CARD_Y_MARGIN,
            );
        }
    }
    // TODO: render tableau
    let cards_in_width = CANVAS_W.get().div_ceil((CARD_WIDTH + COLUMN_MARGIN) as u32) + 1;
    let rightmost_column = leftmost_column + cards_in_width as i64;
    if rightmost_column >= 3 {
        TABLEAU.with_borrow(|tableau| {
            for n in leftmost_column.max(0) as u32..rightmost_column as u32 {
                if n >= 3 {
                    draw_data.push(
                        (n - leftmost_column as u32) as i32 * (CARD_WIDTH + COLUMN_MARGIN) as i32,
                    );
                    draw_data.push(-((CARD_HEIGHT + COLUMN_MARGIN) as i32));
                    draw_data.push(
                        tableau
                            .get(&n)
                            .map(|card| card.as_u8())
                            .unwrap_or(0b10001111) as i32,
                    );
                }
            }
        })
    }

    HELD_HAND.with_borrow(|hand| {
        let [x, y] = MOUSE_POS.get();
        let camera_x_off = CAMERA_X.get() % (CARD_WIDTH + COLUMN_MARGIN);
        append_stack(
            draw_data,
            hand,
            x as i32 + camera_x_off as i32,
            y as i32 + CAMERA_Y.get() as i32,
        );
    });
}

#[wasm_bindgen]
pub fn on_resize(w: u32, h: u32) {
    CANVAS_W.set(w);
    CANVAS_H.set(h);
    RENDER_CONTEXT.with_borrow(|info| {
        let info = info.as_ref().expect_throw("info must exist");
        info.context
            .uniform2ui(Some(&info.canvas_size_uniform), w, h);
        info.context.viewport(0, 0, w as i32, h as i32);
        draw(info);
    });
}

#[wasm_bindgen]
pub fn first_resize(w: u32, h: u32) {
    CAMERA_X.set(-(w as i64 - 6 * (CARD_WIDTH + COLUMN_MARGIN)));
    on_resize(w, h);
} 

#[wasm_bindgen]
pub fn card_tex_loaded(image: JsValue) {
    RENDER_CONTEXT.with_borrow(|info| {
        let info = info.as_ref().expect("info must exist");
        info.context
            .bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&info.card_texture));
        info.context
            .tex_image_2d_with_u32_and_u32_and_html_image_element(
                WebGl2RenderingContext::TEXTURE_2D,
                0,
                WebGl2RenderingContext::RGBA as i32,
                WebGl2RenderingContext::RGBA,
                WebGl2RenderingContext::UNSIGNED_BYTE,
                image.dyn_ref().unwrap_throw(),
            )
            .unwrap_throw();

        info.context.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_S,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
        info.context.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_T,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
        info.context.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MIN_FILTER,
            WebGl2RenderingContext::LINEAR as i32,
        );

        draw(info)
    })
}
