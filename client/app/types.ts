import { Blade, BladeState } from '@tweakpane/core';

interface StreamResponse {
  client_id: number;
  snake_data: SnakeData[];
  food_data: number[][];
}

interface SnakeData {
  uid: number;
  pos: number[];
  vel: { [key: number]: number[] };
  score: number;
  alive: boolean;
  history: number[][];
  fitness: number;
}

declare global {
  interface Window { data: StreamResponse | null; }
}

interface Presets extends BladeState {
  
  n_dims: number; // int
  ttl: number; // int
  width: number; // int
  turn_angle: number; // float
  speed: number; // float
  n_snakes: number; // int
  sigma: number; // float
  alpha: number; // float
  hidden_layer_size: number; // int
}

const defaultPreset: Presets = {
  n_dims: 4,
  ttl: 200,
  width: 20,
  turn_angle: -1,
  speed: 1,
  n_snakes: 20,
  sigma: 0.00001,
  alpha: 0.001,
  hidden_layer_size: 20
}

interface ViewPresets extends BladeState {
  colorMode: string;
  viewBounds: boolean;
}

const defaultViewPreset: ViewPresets = {
  colorMode: "alive",
  viewBounds: true,
}

export type { StreamResponse, SnakeData, Presets, }