export const vertexShaderSource = `#version 300 es

in vec4 aVertexPosition;
in vec4 aVertexColor;
in vec3 aVertexNormal;

uniform mat4 uNormalMatrix;
uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;
uniform sampler2D uPositionTexture;
uniform highp vec3 uAmbientIntensity;
uniform highp vec3 uDirectionalIntensity;
uniform highp vec3 uDirectionalPosition;

out lowp vec4 vColor;
out highp vec3 vLighting;
flat out int id;
out highp vec2 coords;

void main(void) {
    id = gl_InstanceID;

    ivec2 texShape = textureSize(uPositionTexture, 0);
    float texHeight = float(texShape.y);
    float texWidth = float(texShape.x);
    float v = (floor(float(id) / texWidth) + 0.5) / texHeight;
    float u = (mod(float(id), texWidth) + 0.5) / texWidth;
    coords = vec2(u, v);
    vec4 offset = texture(uPositionTexture, coords);

    // Transform + offset
    vec4 worldPos = uModelViewMatrix * aVertexPosition;
    gl_Position = uProjectionMatrix * (worldPos + vec4(offset.rgb, 0.0)); // use 0.0 as w for offset

    // Apply lighting effect
    highp vec3 ambientLight = vec3(0.3, 0.3, 0.3);
    highp vec3 directionalLightColor = vec3(1, 1, 1);
    highp vec3 directionalVector = normalize(vec3(0.85, 0.8, 0.75));
    
    highp vec4 transformedNormal = uNormalMatrix * vec4(aVertexNormal, 1.0);
    
    highp float directional = max(dot(transformedNormal.xyz, normalize(uDirectionalPosition)), 0.0);
    vLighting = uAmbientIntensity + (uDirectionalIntensity * directional);

    // Pass color to fragment shader
    vColor = aVertexColor;
}
`;

export const fragmentShaderSource = `#version 300 es
precision highp float;

in lowp vec4 vColor;
in highp vec3 vLighting;
flat in int id;
out lowp vec4 fragColor;
uniform sampler2D uColorTexture;
in highp vec2 coords;

void main(void) {
    float texHeight = float(textureSize(uColorTexture, 0).y); // expect 2
    float u = (float(id) + 0.5) / texHeight;
    vec4 sampledColor = texture(uColorTexture, coords);

    fragColor = vec4(vLighting, 1.0) * sampledColor;
}
`;
